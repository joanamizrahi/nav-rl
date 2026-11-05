"""RealWorldBackend — NeoVerse-based WorldBackend implementation for Milestone B.

Wraps NeoVerse's pipeline as a Gym-compatible world model backend. Same interface
as MockWorldBackend so SceneEnv doesn't care which backend is plugged in.

Coordinate conventions
----------------------
SceneEnv's world frame (from scene_env.py):
    x = robot forward
    y = robot right
    z = up (gravity)

Reconstructor's world frame (OpenCV / gsplat convention; verified empirically
after the 380492 rasterizer_only test showed y-flipped renders):
    x = right (ground plane)
    y = DOWN (perpendicular to ground, standard CV image-y direction)
    z = forward (camera looks along +z at frame 0)
Our earlier "y = up" guess from the pose diagnostic was wrong-signed — the
diagnostic only told us y was the ground-normal axis, not which direction.

Robot's local frame (what SceneEnv._advance_pose assumes):
    local +x = forward direction the robot is facing

Camera local frame (NeoVerse convention, standard CV):
    local +x = right
    local +y = down
    local +z = forward

So `render(pose_scene)` has to convert:
    pose_scene (robot-to-world in scene frame)
    -> pose_recon (camera-to-world in reconstructor frame)

...via two fixed 4x4 rotations built into the constants below.

Scale
-----
Reconstructor units are NORMALIZED per clip, not meters. See REWARD.md and
OPTIONS.md. This backend accepts scene-frame poses in the same "recon units"
convention and does NOT apply a scale factor. Reward function and step sizes
in SceneEnvConfig should be tuned assuming the world is ~1 unit ≈ few real
meters. Deploy-time scale calibration is a separate problem.

Coverage
--------
The Gaussians only exist where the source video's camera saw them. If the RL
policy chooses a pose FAR from the source trajectory, the rasterizer will
return mostly-holey output that the diffusion tries to inpaint. Quality
degrades gracefully as the policy strays from the reconstructed volume.
"""
from __future__ import annotations
import os as _os
_S = _os.environ.get("SCRATCH_ROOT", "/scratch/m000204-pm07b/joana")  # 2026-11-05: Cycle 7 scratch; override with SCRATCH_ROOT

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate conversion constants
# ---------------------------------------------------------------------------

# Scene world -> Reconstructor world.
# Scene:  x=fwd,   y=right, z=up
# Recon:  x=right, y=DOWN,  z=fwd    (OpenCV convention)
# scene_y (right) ->  recon_x (right)   row 0
# scene_z (up)    -> -recon_y (down)    row 1  <-- flipped after 380492 upside-down bug
# scene_x (fwd)   ->  recon_z (fwd)     row 2
R_SCENE_TO_RECON = np.array([
    [0.0, 1.0,  0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0,  0.0],
], dtype=np.float32)

# Robot local -> Camera local (takes a point expressed in robot frame -> camera frame).
# robot_x (fwd)  -> camera_z (fwd)
# robot_y (right)-> camera_x (right)
# robot_z (up)   -> -camera_y (up = -down)
R_ROBOT_TO_CAM = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
], dtype=np.float32)

# Camera local -> Robot local (inverse of above; for rotation matrices, transpose).
# We APPLY this on the right of pose_scene (r2w) to build c2w:
#     c2w = r2w @ R_CAM_TO_ROBOT
# Because c2w takes camera-frame points to world, and we want:
#     point_world = c2w @ point_camera_frame
#                 = r2w @ R_CAM_TO_ROBOT @ point_camera_frame
#                 = r2w @ point_robot_frame
#                 = point_world  ✓
R_CAM_TO_ROBOT = R_ROBOT_TO_CAM.T


def _move_tree_to(x, device):
    """Recursively move tensors inside a dict / list / tuple / tensor to `device`."""
    import torch
    if torch.is_tensor(x):
        return x.detach().to(device)
    if isinstance(x, dict):
        return {k: _move_tree_to(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        moved = [_move_tree_to(v, device) for v in x]
        return type(x)(moved)
    return x


def _pose_scene_to_recon(pose_scene: np.ndarray, camera_height_scene: float = 0.0) -> np.ndarray:
    """Convert a 4x4 robot-to-scene-world pose into a 4x4 camera-to-recon-world pose.

    Steps:
      1. Optionally lift the camera above the robot's position by `camera_height_scene`
         (in scene world coords, along the scene z-up axis).
      2. Apply the robot->camera local rotation on the RIGHT (rotates the local frame).
      3. Apply the scene->recon world rotation on the LEFT (rotates the world frame).
    """
    T_lift = np.eye(4, dtype=np.float32)
    T_lift[2, 3] = camera_height_scene    # scene z is up

    # Build the two 4x4 rotations
    T_c2r_local = np.eye(4, dtype=np.float32)
    T_c2r_local[:3, :3] = R_CAM_TO_ROBOT           # camera-local axes -> robot-local axes
    T_sr = np.eye(4, dtype=np.float32)
    T_sr[:3, :3] = R_SCENE_TO_RECON

    # pose_scene: robot-to-scene-world (r2w).
    # camera-to-scene-world (c2w):
    #     c2w = T_lift @ pose_scene @ T_c2r_local
    #     (T_lift adjusts the world position of the camera above the robot,
    #      T_c2r_local converts a point expressed in camera-local axes into
    #      robot-local axes before r2w maps it to world.)
    # camera-to-recon-world:
    #     c2w_recon = T_sr @ c2w_scene
    return T_sr @ (T_lift @ pose_scene @ T_c2r_local)


# ---------------------------------------------------------------------------
# Backend config
# ---------------------------------------------------------------------------

@dataclass
class RealWorldBackendConfig:
    # Where to find the source video for each scene_id (scene_id -> mp4 path).
    scene_video_paths: dict = field(default_factory=dict)

    # Where to put the robot at reset — defaults to identity (origin in scene frame).
    scene_start_poses: dict = field(default_factory=dict)

    # Where the goal is in scene frame per scene_id.
    scene_goals: dict = field(default_factory=dict)

    # Camera mounted this high above the robot's body-origin in SceneEnv's world
    # frame (along scene +z, up).
    #
    # DEFAULT = 0.0: identity pose_scene puts the camera exactly where the
    # SOURCE VIDEO's frame-0 camera was. Simplest, matches what the reconstructor
    # actually gives us. Any real-robot calibration (Go2 camera is ~0.4m above
    # its feet in real world) happens at deploy time by mapping physical actions
    # to sim actions -- not by trying to compensate here.
    #
    # A non-zero value would try to invent a "body position" below the camera,
    # which never existed in the source video and can produce off-manifold renders.
    camera_height_scene: float = 0.0

    # Rendering config
    H: int = 336
    W: int = 560
    num_frames: int = 81       # frames sampled from source video for reconstruction
    # STATIC reconstruction (2026-09-07, Joana: "why not use all the frames"):
    # with is_static=True every Gaussian of every source frame is CONSTANT
    # (timestamp -1) and renders from any pose at any time, voxel-pruned
    # across frames. With False (the default so far) WorldMirror classifies
    # Gaussians by predicted velocity against dynamic_threshold=0, so NONE
    # are constant and each view draws only the nearest source frame's
    # Gaussians (plus one neighbour): striped alpha off the walk, and an
    # abrupt switch every time the nearest frame changes.
    static_scene: bool = False
    static_movers: str = ""            # "12,13": classes kept per-frame in static mode (see the setter before reconstruct)

    # Diffusion config
    use_lora: bool = True      # 4-step distilled LoRA (fast); False = 50 steps
    cfg_scale: float = 1.0
    prompt: str = ("A smooth video with complete scene content. Inpaint any missing "
                   "regions or margins naturally to match the surrounding scene.")
    negative_prompt: str = ""

    # Rendering mode:
    #   "rasterizer_plus_diffusion" (DEFAULT) — run 4-step diffusion on top of the
    #       raw Gaussian rasterization. Clean output, ~5x slower per step. Video
    #       prior may hallucinate small dynamics (parked cars fake-driving on
    #       driving.mp4-style scenes). Less problematic on natural / static
    #       scenes like RUGD trails.
    #   "rasterizer_only" — raw rasterizer, no diffusion. Deterministic and cheap
    #       BUT holey outside the reconstructed volume — likely too bad for RL
    #       observations (holes as big as the visible area). Retained for tests.
    render_mode: str = "rasterizer_plus_diffusion"

    # Model paths
    model_path: str = _S + "/NeoVerse/models"
    reconstructor_path: str = _S + "/NeoVerse/models/NeoVerse/reconstructor.ckpt"


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------

class RealWorldBackend:
    """WorldBackend implementation that renders through NeoVerse's pipeline.

    Attributes match the WorldBackend Protocol from scene_env.py:
        H, W: render resolution
        load_scene(scene_id): reconstruct + cache Gaussians for that scene
        render(pose_scene): return (rgb, K, w2c) at the requested robot pose
        start_pose(scene_id): initial robot pose (identity by default)
        goal_position(scene_id): goal in scene world coords
    """

    def __init__(self, cfg: RealWorldBackendConfig = RealWorldBackendConfig()):
        self.cfg = cfg
        self.H = cfg.H
        self.W = cfg.W
        self._pipe = None                       # lazy-loaded, full pipeline (reconstructor + diffusion)
        self._reconstructor = None              # lazy-loaded, reconstructor only (memory-efficient)
        self._current_scene_id: Optional[str] = None
        self._cache = {}                         # scene_id -> per-scene state

    # ------------ WorldBackend API ------------

    def load_scene(self, scene_id: str) -> None:
        """Reconstruct the source video (once per scene) and cache Gaussians + K.

        Only loads the reconstructor here (cheap ~2 GB VRAM). The full pipeline
        (with diffusion) is deferred until the first render() call in
        rasterizer_plus_diffusion mode — avoids blowing memory during
        reconstruction.
        """
        if scene_id in self._cache:
            self._current_scene_id = scene_id
            return
        self._ensure_reconstructor_loaded()

        video_path = self.cfg.scene_video_paths.get(scene_id)
        if not video_path:
            raise ValueError(f"no video path configured for scene_id={scene_id}")

        cached = self._reconstruct_scene(video_path)
        self._cache[scene_id] = cached
        self._current_scene_id = scene_id

    def start_pose(self, scene_id: str) -> np.ndarray:
        return self.cfg.scene_start_poses.get(
            scene_id, np.eye(4, dtype=np.float32)
        ).copy()

    def goal_position(self, scene_id: str) -> np.ndarray:
        # If not configured, default goal is "3 units forward in scene x" (a few meters).
        return self.cfg.scene_goals.get(
            scene_id, np.array([3.0, 0.0, 0.0], dtype=np.float32)
        ).copy()

    def render(self, pose_scene: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (rgb, K, w2c) after rasterize + diffuse at the requested pose.

        pose_scene: (4, 4) robot-to-world matrix in SceneEnv's world frame.
        """
        scene = self._cache.get(self._current_scene_id)
        if scene is None:
            raise RuntimeError("call load_scene() first")

        # Convert robot-to-scene-world -> camera-to-recon-world.
        pose_recon = _pose_scene_to_recon(
            pose_scene.astype(np.float32),
            camera_height_scene=self.cfg.camera_height_scene,
        )
        rgb, K, w2c = self._rasterize_and_diffuse(scene, pose_recon)
        return rgb, K, w2c

    # ------------ heavy lifting (deferred to NeoVerse) ------------

    def _ensure_reconstructor_loaded(self):
        """Load JUST the reconstructor. Cheap (~2 GB VRAM). Sufficient for
        load_scene()'s Gaussian caching step."""
        if self._reconstructor is not None:
            return
        import torch
        from diffsynth.utils import ModelConfig
        from diffsynth.models import ModelManager
        self._torch = torch

        print(f"[RealWorldBackend] loading reconstructor only from {self.cfg.reconstructor_path} ...", flush=True)
        mm = ModelManager()
        cfg = ModelConfig(path=self.cfg.reconstructor_path, offload_device="cuda")
        cfg.download_if_necessary()
        mm.load_model(cfg.path, device="cuda", torch_dtype=torch.bfloat16)
        self._reconstructor = mm.fetch_model("reconstructor")

    def _ensure_pipe_loaded(self):
        """Load the FULL pipeline (reconstructor + diffusion). ~40+ GB VRAM.
        Only needed for rasterizer_plus_diffusion render mode."""
        if self._pipe is not None:
            return
        import torch
        import gc
        from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
        self._torch = torch

        lora_path = None
        if self.cfg.use_lora:
            lora_path = str(Path(self.cfg.model_path) /
                            "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors")

        # Free our separately-loaded reconstructor before pipe load. The pipe's
        # from_pretrained loads its own reconstructor internally; keeping ours
        # around means two copies on GPU. We swap over to the pipe's copy below.
        if self._reconstructor is not None:
            del self._reconstructor
            self._reconstructor = None
            gc.collect()
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            print(f"[RealWorldBackend] freed pre-existing reconstructor. VRAM free {free/1e9:.1f} GB", flush=True)

        print(f"[RealWorldBackend] loading NeoVerse pipeline from {self.cfg.model_path} ...", flush=True)
        self._pipe = WanVideoNeoVersePipeline.from_pretrained(
            local_model_path=self.cfg.model_path,
            reconstructor_path=self.cfg.reconstructor_path,
            lora_path=lora_path, lora_alpha=1.0,
            device="cuda", torch_dtype=torch.bfloat16,
        )
        # Reconstructor is inside the pipe; use pipe's copy for consistency.
        self._reconstructor = self._pipe.reconstructor
        free, _ = torch.cuda.mem_get_info()
        print(f"[RealWorldBackend] pipe loaded. VRAM free {free/1e9:.1f} GB", flush=True)

    def _reconstruct_scene(self, video_path: str) -> dict:
        """Run the reconstructor once and cache Gaussians + K + first-frame pose.

        Uses `self._reconstructor` directly (loaded lazily), not the full pipe.
        This lets load_scene() run before we've paid the ~30GB VRAM cost of the
        Wan diffusion weights.
        """
        import torch
        from torchvision.transforms import functional as F
        from diffsynth.utils.auxiliary import load_video

        cfg = self.cfg
        reconstructor = self._reconstructor
        device = next(reconstructor.parameters()).device
        dtype = next(reconstructor.parameters()).dtype

        print(f"[RealWorldBackend] loading source video: {video_path}", flush=True)
        images = load_video(
            video_path, cfg.num_frames,
            resolution=(cfg.W, cfg.H), resize_mode="center_crop", static_scene=False,
        )
        views = {
            "img": torch.stack([F.to_tensor(im)[None] for im in images], dim=1).to(device),
            "is_target": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
            "is_static": (torch.ones if bool(getattr(cfg, "static_scene", False)) else torch.zeros)((1, len(images)), dtype=torch.bool, device=device),
            "timestamp": torch.arange(0, len(images), dtype=torch.int64, device=device).unsqueeze(0),
        }
        print(f"[RealWorldBackend] running reconstructor ({'STATIC: all frames constant' if bool(getattr(cfg, 'static_scene', False)) else 'dynamic: per-frame Gaussians'}) ...", flush=True)
        free_before, _ = torch.cuda.mem_get_info()
        # torch.no_grad() is CRITICAL: WorldMirror is a 40-layer transformer.
        # Without it, the full forward computation graph is kept alive for
        # backprop (~50-70 GB of activations at 82 frames), which then can't be
        # reclaimed by empty_cache(). This is what blew our VRAM before Wan
        # could load.
        # STATIC MOVERS (2026-09-07): with static_scene, Gaussians labelled with
        # these classes (12 person, 13 vehicle) stay per-frame instead of being
        # fused into the constant set, so a pedestrian is drawn once, at their
        # own frame, not as a trail over every view. Must match the label
        # head's training (v31/v32 = 12,13; v26/v30 = none).
        _mv = tuple(int(v) for v in str(getattr(cfg, "static_movers", "") or "").split(",") if v.strip())
        _rast = reconstructor.gs_renderer   # owns separate_splats / _classify_gaussians; .rasterizer only draws
        if tuple(getattr(_rast, "dynamic_label_ids", ())) != _mv:
            _rast.dynamic_label_ids = _mv
        if _mv:
            print(f"[RealWorldBackend] static movers: classes {_mv} stay per-frame", flush=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            predictions = reconstructor(views, is_inference=True, use_motion=False)
        free_after, _ = torch.cuda.mem_get_info()
        print(f"[RealWorldBackend] reconstructor VRAM: free {free_before/1e9:.1f} -> {free_after/1e9:.1f} GB", flush=True)

        # Move all cached tensors to CPU so we can free VRAM before loading the
        # ~30GB Wan diffusion pipeline. During render() we'll move them back to
        # GPU on demand. Skipped if we're in rasterizer_only mode -- no need for
        # the round trip.
        # CRITICAL: don't hold any local Python name that references the GPU
        # tensor after we've made the CPU copy. Otherwise the GPU tensor stays
        # alive and empty_cache() can't reclaim its memory.
        move_to_cpu = (self.cfg.render_mode == "rasterizer_plus_diffusion")
        if move_to_cpu:
            cache = {
                "gaussians": _move_tree_to(predictions["splats"], "cpu"),
                "K": predictions["rendered_intrinsics"][0].detach().cpu(),
                "cam2world": predictions["rendered_extrinsics"][0].detach().cpu(),
                "timestamps": predictions["rendered_timestamps"][0].detach().cpu(),
                "views": _move_tree_to(views, "cpu"),
            }
            del predictions, views
            # Also free any reconstructor intermediate state on GPU.
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print(f"[RealWorldBackend] moved scene cache to CPU. VRAM free {free / 1e9:.1f} GB / {total / 1e9:.1f} GB", flush=True)
        else:
            cache = {
                "gaussians": predictions["splats"],
                "K": predictions["rendered_intrinsics"][0],
                "cam2world": predictions["rendered_extrinsics"][0],
                "timestamps": predictions["rendered_timestamps"][0],
                "views": views,
            }
        return cache

    def _rasterize_and_diffuse(self, scene: dict, pose_recon: np.ndarray):
        """Rasterize the cached Gaussians at pose_recon, optionally diffuse.

        pose_recon: (4, 4) camera-to-recon-world matrix (numpy).

        For `rasterizer_only` mode we use only `self._reconstructor` (which was
        loaded during load_scene). For `rasterizer_plus_diffusion` mode we
        lazily load the full pipe here — deferring the ~30GB Wan diffusion
        weights until absolutely needed.
        """
        import torch
        from diffsynth.utils.auxiliary import homo_matrix_inverse

        reconstructor = self._reconstructor
        device = next(reconstructor.parameters()).device

        # If the scene was moved to CPU (diffusion mode) after reconstruction,
        # move it back to GPU now that we're about to rasterize / diffuse.
        # Rasterizer_only mode leaves the scene on GPU already, so this is a no-op there.
        needs_move_back = self.cfg.render_mode == "rasterizer_plus_diffusion"
        if needs_move_back and torch.is_tensor(scene["K"]) and scene["K"].device.type == "cpu":
            scene = _move_tree_to(scene, device)

        N = len(scene["timestamps"])

        # Broadcast our single requested pose to N frames (matches the reconstructor's timestamp layout).
        fixed_c2w = torch.from_numpy(pose_recon).to(device=device, dtype=scene["cam2world"].dtype).unsqueeze(0).repeat(N, 1, 1)
        fixed_w2c = homo_matrix_inverse(fixed_c2w)
        K_rep = scene["K"][0:1].repeat(N, 1, 1)

        target_rgb, target_depth, target_alpha = reconstructor.gs_renderer.rasterizer.forward(
            scene["gaussians"],
            render_viewmats=[fixed_w2c], render_Ks=[K_rep],
            render_timestamps=[scene["timestamps"]],
            sh_degree=0, width=self.W, height=self.H,
        )

        if self.cfg.render_mode == "rasterizer_only":
            # Fast path: raw rasterizer output, no diffusion, no video-prior
            # hallucinations. Take the first rendered frame as the observation.
            rgb_frame = (target_rgb[0, 0].detach().clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
            K_np = scene["K"][0].detach().cpu().float().numpy()
            w2c_np = fixed_w2c[0].detach().cpu().float().numpy()
            return rgb_frame, K_np, w2c_np

        # Slow path: full 4-step diffusion. Runs on all N frames at once and takes
        # the FIRST rendered frame as the observation. Cleaner output but has
        # video-prior hallucinations (see freeze_camera_diffuse.py test on driving.mp4).
        # Lazy-load the full pipe now — deferred from load_scene() to keep memory low.
        self._ensure_pipe_loaded()
        pipe = self._pipe
        target_mask = (target_alpha > 1.0).float()
        wrapped_data = {
            "source_views": scene["views"],
            "target_rgb": target_rgb, "target_depth": target_depth, "target_mask": target_mask,
            "target_poses": fixed_c2w.unsqueeze(0), "target_intrs": K_rep.unsqueeze(0),
        }
        with torch.no_grad():
            generated = pipe(
                prompt=self.cfg.prompt, negative_prompt=self.cfg.negative_prompt,
                seed=0, rand_device=device,
                height=self.H, width=self.W, num_frames=N,
                cfg_scale=self.cfg.cfg_scale, num_inference_steps=4 if self.cfg.use_lora else 50,
                tiled=False, **wrapped_data,
            )
        rgb_frame = np.array(generated[0])
        K_np = scene["K"][0].detach().cpu().float().numpy()
        w2c_np = fixed_w2c[0].detach().cpu().float().numpy()
        return rgb_frame, K_np, w2c_np
