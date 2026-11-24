"""Live semantic-diffusion backend: per-action generation, no cache.

The policy queries the generative model along its own motion instead of a
pre-rendered grid (design decision 2026-08-21). Every env.step renders ONE
minimal diffusion call — a 5-frame clip whose
frames are the robot's 4 previous poses + the current pose (the video model's
4k+1 minimum, filled with true history rather than padding), last frame = the
observation. The SAME call's semantic half feeds the reward, so the ungated
recipe trains against the diffusion model's labels at the exact pose, with no
grid discretization.

Differences from the old live path (real_backend._rasterize_and_diffuse):
  - loads the SEMANTIC pipeline (v10 checkpoint: expansion v2 + LoRA + reader
    head) — the old path loaded the vanilla model and had no diffused labels;
  - renders k requested poses, not 1 pose broadcast to 81 (30 s -> target ~3 s);
  - scene stays resident on GPU (no CPU<->GPU shuttle per call);
  - nothing is written to disk.

Scene time advances with the robot (per-pose nearest source frame), so a
dynamic reconstruction (SCAND) renders moving pedestrians — something caches
structurally cannot do (they freeze time per sweep).

Speed instrumentation: self.last_timings holds per-phase seconds for the most
recent render; scripts/live_benchmark.py builds the decision table from it.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .real_calibrated import CalibratedRealWorldBackend


def cold_history(pose_recon: np.ndarray, scene: dict, k: int, use_recorded: bool = True) -> list:
    """Synthesize a straight walk-in ending at pose_recon.

    Five IDENTICAL poses (the old cold start) are out-of-distribution for the
    video model — with no motion context its generic prior takes over and it
    renders faces/sheep/buildings for the first steps of every episode
    (measured 2026-08-29 on drive previews; the policy's every reset began
    with hallucinations, teaching it to distrust vision). Back-extrapolating
    along the camera's forward axis, stepped by the scene's own median
    inter-frame spacing, gives the model a coherent approach clip instead.
    """
    c2w_all = scene["cam2world"]
    c2w_np = c2w_all.detach().cpu().numpy() if hasattr(c2w_all, "detach") else np.asarray(c2w_all)
    src = c2w_np[:, :3, 3]
    steps = np.linalg.norm(np.diff(src, axis=0), axis=1)
    step = float(np.median(steps)) if len(steps) else 0.05
    fwd = pose_recon[:3, 2].astype(np.float64)
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])

    # 2026-09-16: the RECORDED frames behind the spawn are the true approach.
    # Extrapolating straight back along the camera axis put the approach poses
    # outside the reconstruction at the start of a walk (quad2_04: garbage
    # first generations until real history existed) and off the path at bends.
    # Take the k-1 recorded frames before the nearest frame, shifted by the
    # spawn's offset from that frame so the clip ends exactly at pose_recon;
    # fall back to extrapolation for the entries that do not exist (frame < k-1).
    if bool(use_recorded) and len(src) >= 2:
        d = np.linalg.norm(src - pose_recon[:3, 3][None, :], axis=1)
        i_star = int(np.argmin(d))
        offset = pose_recon[:3, 3].astype(np.float64) - src[i_star]
        hist = []
        for j in range(k - 1, 0, -1):
            i = i_star - j
            if i >= 0:
                p = c2w_np[i].astype(np.float64).copy()
                p[:3, 3] = src[i] + offset
            else:
                p = pose_recon.astype(np.float64).copy()
                p[:3, 3] = pose_recon[:3, 3] - fwd * step * j
            hist.append(p.astype(np.float32))
        hist.append(pose_recon.astype(np.float32))
        return hist

    hist = []
    for i in range(k - 1, 0, -1):
        p = pose_recon.copy()
        p[:3, 3] = pose_recon[:3, 3] - (fwd * step * i).astype(np.float32)
        hist.append(p.astype(np.float32))
    hist.append(pose_recon.astype(np.float32))
    return hist


# ---- recording-like context poses (INTERP, 2026-11-24) -------------------------------
# The video model was fine-tuned on walking clips: ~0.23 m and a few degrees between frames.
# A turning robot's last five poses are up to 17 deg apart, a camera whip the model never saw,
# and the last frame (the observation) comes out worse. With interp_history on, the four
# CONTEXT poses are rebuilt along the robot's own recent path, spaced and turned like the
# recording; the last pose stays exactly the current one. Coverage is unaffected (it reads
# only the last pose).
INTERP_PATH_LEN = 16          # robot poses kept per history (the path the context is cut from)


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _rot_towards(Ra: np.ndarray, Rb: np.ndarray, t: float) -> np.ndarray:
    """Rotation a fraction t of the way from Ra to Rb (geodesic)."""
    Rr = Ra.T @ Rb
    ang = _rot_angle(np.eye(3), Rr)
    if ang < 1e-9 or t <= 0.0:
        return Ra.copy()
    if t >= 1.0:
        return Rb.copy()
    w = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]])
    n = np.linalg.norm(w)
    if n < 1e-9:                                   # ~180 deg: any perpendicular axis
        axis = np.linalg.svd(Rr - np.eye(3))[2][-1]
    else:
        axis = w / n
    a = ang * t
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return Ra @ (np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * (Kx @ Kx))


def recording_spacing(scene: dict) -> tuple:
    """(median step between recorded frames in recon units, 90th-percentile turn per frame in rad), cached."""
    if "_interp_spacing" in scene:
        return scene["_interp_spacing"]
    c2w = scene["cam2world"]
    c2w = c2w.detach().cpu().numpy() if hasattr(c2w, "detach") else np.asarray(c2w)
    c2w = c2w.astype(np.float64)
    steps = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    turns = [_rot_angle(c2w[i, :3, :3], c2w[i + 1, :3, :3]) for i in range(len(c2w) - 1)]
    # turn cap: the 90th percentile of the recording's per-frame turns, i.e. how fast the
    # recording turns at its corners (the median is mostly straight walking and would leave the
    # context of a turning robot nearly un-turned). Floor 1 deg. (Joana, 2026-11-24)
    out = (float(np.median(steps)) if len(steps) else 0.05,
           max(float(np.percentile(turns, 90)) if turns else 0.0, np.radians(1.0)))
    scene["_interp_spacing"] = out
    print(f"[interp] recording spacing: step {out[0]:.4f} recon units, turn cap {np.degrees(out[1]):.2f} deg per frame (p90)",
          flush=True)
    return out


def respace_history(path: list, scene: dict, k: int) -> list:
    """k poses ending at path[-1], walked back along `path` at the recording's spacing, each
    context pose turned at most j x (recording's per-frame turn) away from the current heading."""
    step, turn = recording_spacing(scene)
    cur = np.asarray(path[-1], dtype=np.float64)
    pts = [np.asarray(p, dtype=np.float64)[:3, 3] for p in reversed(path)]       # newest first
    rots = [np.asarray(p, dtype=np.float64)[:3, :3] for p in reversed(path)]
    # extend beyond the oldest pose so every target lies on the polyline: continue the oldest
    # segment's direction, or (no motion at all, e.g. turning in place) straight back along the
    # current camera axis -- the cold_history fallback
    d = pts[-1] - pts[-2] if len(pts) >= 2 else np.zeros(3)
    if np.linalg.norm(d) < 1e-9:
        d = -cur[:3, 2]
    d = d / max(np.linalg.norm(d), 1e-9)
    pts.append(pts[-1] + d * step * (k + 1)); rots.append(rots[-1])
    seg = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for j in range(k - 1, 0, -1):
        s = j * step
        i = int(np.searchsorted(cum, s, side="left")) - 1
        i = min(max(i, 0), len(seg) - 1)
        while i < len(seg) - 1 and seg[i] < 1e-12:
            i += 1
        f = 0.0 if seg[i] < 1e-12 else float(np.clip((s - cum[i]) / seg[i], 0.0, 1.0))
        pos = pts[i] + f * (pts[i + 1] - pts[i])
        Rp = _rot_towards(rots[i], rots[i + 1], f)
        ang = _rot_angle(cur[:3, :3], Rp)
        cap = j * turn
        R = _rot_towards(cur[:3, :3], Rp, 1.0 if ang <= cap else cap / ang)
        p = cur.copy(); p[:3, :3] = R; p[:3, 3] = pos
        out.append(p.astype(np.float32))
    out.append(cur.astype(np.float32))
    return out


class LiveDiffusedBackend(CalibratedRealWorldBackend):
    """Per-action live generation with the semantic diffusion pipeline."""

    def __init__(self, cfg, checkpoint: str, live_frames: int = 5,
                 alpha_gate: bool = False, alpha_gate_tau: float = 0.5,
                 num_classes: int = 14,
                 lora_rank: int = 8,
                 lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2",
                 raster_obs: bool = False):
        # Any mode other than "rasterizer_plus_diffusion" keeps the scene
        # resident on GPU in _reconstruct_scene (both parent and calibrated
        # override) — exactly what we want here.
        cfg.render_mode = "live_diffusion"
        super().__init__(cfg)
        self._sem_checkpoint = checkpoint
        self.live_frames = int(live_frames)
        self._alpha_gate = bool(alpha_gate)
        # Gate threshold. 0.5 was inherited and never validated; the sweep on
        # 2026-09-01 (gnd_AUw360, 32 renders) showed alpha is BIMODAL — 24% of
        # pixels below 0.1, 48% above 0.9, a shallow trough between — and that
        # every phantom crash is already removed at tau=0.1, so everything
        # above that is pure cost. Default stays 0.5 so old runs reproduce.
        self._alpha_gate_tau = float(alpha_gate_tau)
        # Raster-observation ablation (2026-09-06, Joana's inconsistency
        # question): the policy SEES the reconstruction's own raster and the
        # reward reads the splat labels, so observation and map put the edge
        # in the same place. No diffusion call per step. Diagnostic only --
        # the raster does not look like a camera, so this policy is not the
        # one that goes on the robot.
        self.raster_obs = bool(raster_obs)
        if not hasattr(self, "_pipe"):
            self._pipe = None          # raster mode reads it without ever loading it
        if self.raster_obs:
            print("[LiveDiffusedBackend] RASTER OBSERVATIONS: the policy sees the "
                  "splat raster and the reward reads the splat labels; the "
                  "diffusion model is never loaded", flush=True)
        self._num_classes = int(num_classes)
        self._lora_rank = int(lora_rank)
        self._lora_targets = lora_target_modules
        self._pose_hist: list = []            # recon-frame c2w, newest last
        self.interp_history = False           # set by the caller (INTERP); see respace_history
        self._pending_labels: Optional[np.ndarray] = None
        self.last_timings: dict = {}

    # ---------- semantic pipeline (loaded once, resident) ----------

    def _ensure_semantic_pipe(self):
        if getattr(self, "_pipe", None) is not None:
            return
        import gc
        import torch
        from pathlib import Path
        from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
        from diffsynth.utils.semantics import (
            expand_dit_for_semantics_v2, expand_control_branch_for_semantics_v2,
            set_active_palette,
        )
        from inference_semantic import (
            _inject_lora_for_finetune, _load_finetune_checkpoint,
        )
        self._torch = torch

        # Same discipline as the parent's _ensure_pipe_loaded: drop any
        # standalone reconstructor before the pipe brings its own copy.
        if self._reconstructor is not None:
            del self._reconstructor
            self._reconstructor = None
            gc.collect()
            torch.cuda.empty_cache()

        lora_path = None
        if self.cfg.use_lora:
            lora_path = str(Path(self.cfg.model_path) /
                            "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors")
        print(f"[LiveDiffusedBackend] loading SEMANTIC pipeline "
              f"(ckpt={self._sem_checkpoint}) ...", flush=True)
        pipe = WanVideoNeoVersePipeline.from_pretrained(
            local_model_path=self.cfg.model_path,
            reconstructor_path=self.cfg.reconstructor_path,
            lora_path=lora_path, lora_alpha=1.0,
            device="cuda", torch_dtype=torch.bfloat16,
        )
        pipe.semantic_channels = 16
        pipe.semantic_x0_prediction = True
        _pal = int(getattr(self.cfg, "sem_palette_version", 1))
        set_active_palette(self._num_classes, _pal)
        print(f"[LiveDiffusedBackend] active palette: v{_pal} "
              f"({self._num_classes} classes)", flush=True)
        expand_dit_for_semantics_v2(pipe.dit, extra=16)
        if pipe.control_branch is not None:
            expand_control_branch_for_semantics_v2(pipe.control_branch, extra=16)
        _inject_lora_for_finetune(pipe, rank=self._lora_rank,
                                  target_modules=self._lora_targets.split(","))
        _load_finetune_checkpoint(pipe, self._sem_checkpoint)
        # peft injection leaves fresh LoRA modules in train mode, flipping the
        # pipeline's is_training property — the 81-frame source clip is then
        # VAE-encoded as input latents, which shape-crashes any num_frames!=81
        # call (cache_gen never hits this: its calls always match source length).
        pipe.eval()
        self._pipe = pipe
        self._reconstructor = pipe.reconstructor
        free, _ = torch.cuda.mem_get_info()
        print(f"[LiveDiffusedBackend] semantic pipe ready. "
              f"VRAM free {free/1e9:.1f} GB", flush=True)

    def reset_history(self) -> None:
        """Drop the single-robot frame history at an episode reset (evals)."""
        self._pose_hist = []
        self.hist_resets_explicit = int(getattr(self, "hist_resets_explicit", 0)) + 1
        if self.hist_resets_explicit in (1, 100, 1000):
            print(f"[LiveDiffusedBackend] frame history cleared at reset: {self.hist_resets_explicit} times so far "
                  f"(jump-rule resets {int(getattr(self, 'hist_resets_jump', 0))})", flush=True)

    def load_scene(self, scene_id: str) -> None:
        # Pipe (and therefore the reconstructor used for scene building) must
        # exist before reconstruction so there is exactly one copy on GPU.
        # Raster mode never diffuses: the parent's reconstructor-only load
        # (~2 GB, seconds) replaces the ~40 GB pipe load.
        if not self.raster_obs:
            self._ensure_semantic_pipe()
        super().load_scene(scene_id)
        self._pose_hist = []

    def _recon(self):
        """The reconstructor that owns the rasterizer: the pipe's copy when the
        pipe is loaded, the standalone one in raster mode."""
        pipe = getattr(self, "_pipe", None)
        if pipe is not None:
            return pipe.reconstructor
        if self._reconstructor is None:
            raise RuntimeError("no reconstructor loaded: call load_scene() first")
        return self._reconstructor

    # ---------- per-step live render ----------

    def _rasterize_and_diffuse(self, scene: dict, pose_recon: np.ndarray):
        # direct import, not a handle stored by _ensure_semantic_pipe: the
        # benchmark hands later backends an already-built pipe, skipping setup
        import torch
        from diffsynth.utils.auxiliary import homo_matrix_inverse
        from inference_semantic import (
            _make_dual_decode, _sem_video_to_labels_and_colorized,
        )
        pipe = self._pipe
        recon = self._recon()
        device = next(recon.parameters()).device
        t = {}
        t0 = time.perf_counter()

        # Rolling pose history. A jump (reset/teleport) restarts the clip so
        # the conditioning never spans a discontinuity.
        pose_recon = pose_recon.astype(np.float32)
        if self._pose_hist:
            # in METRES (recon units are normalized; see vec_live_env._robot_hist)
            _cal = self._calib.get(self._current_scene_id)
            _m_per_unit = float(getattr(_cal, "scale", 1.0) or 1.0)
            jump_m = float(np.linalg.norm(self._pose_hist[-1][:3, 3] - pose_recon[:3, 3])) * _m_per_unit
            if jump_m > 0.5:
                self._pose_hist = []
                self.hist_resets_jump = int(getattr(self, "hist_resets_jump", 0)) + 1
        if not self._pose_hist:
            self._pose_hist = cold_history(pose_recon, scene, self.live_frames)
        else:
            self._pose_hist.append(pose_recon)
            self._pose_hist = self._pose_hist[-(INTERP_PATH_LEN if self.interp_history else self.live_frames):]
        _clip = (respace_history(self._pose_hist, scene, self.live_frames) if self.interp_history
                 else self._pose_hist)
        k = len(_clip)

        c2w = torch.from_numpy(np.stack(_clip)).to(
            device=device, dtype=scene["cam2world"].dtype)          # [k,4,4]
        w2c = homo_matrix_inverse(c2w)
        K_rep = scene["K"][0:1].repeat(k, 1, 1)

        # Scene time per pose: nearest source-camera position — the clock
        # advances as the robot moves (dynamic scenes render motion).
        src_pos = scene["cam2world"][:, :3, 3]                      # [N,3]
        d = torch.cdist(c2w[:, :3, 3].float(), src_pos.float())
        t_idx = d.argmin(dim=1)
        target_ts = scene["timestamps"][t_idx]

        raster = recon.gs_renderer.rasterizer
        t_last = int(t_idx[-1])
        gs_render = self._render_gaussians(scene, t_last)      # fusion window (or all)
        target_rgb, target_depth, target_alpha = raster.forward(
            gs_render, render_viewmats=[w2c], render_Ks=[K_rep],
            render_timestamps=[target_ts], sh_degree=0,
            width=self.W, height=self.H)
        sem_probs, _, _ = raster.forward(
            gs_render, render_viewmats=[w2c], render_Ks=[K_rep],
            render_timestamps=[target_ts], sh_degree=0,
            width=self.W, height=self.H, feature="labels")
        # coverage the gate reads: from the coverage window when set, else the
        # render's own alpha (full-scene alpha saturates in static mode)
        cov_win = self._window_coverage(scene, w2c[-1], K_rep[-1], target_ts[-1], t_last)
        target_semantic = sem_probs.argmax(dim=-1).to(torch.long)
        target_mask = (target_alpha > 1.0).float()   # conditioning threshold (cache-gen parity)
        t["raster"] = time.perf_counter() - t0

        if self.raster_obs:
            rgb_frame = (target_rgb[0, -1].detach().clamp(0, 1).float().cpu().numpy()[..., :3]
                         * 255.0).astype(np.uint8)
            lab_last = target_semantic[0, -1].detach().cpu().numpy().astype(np.int8)
            alpha_last = (target_alpha[0, -1].detach().float().cpu().numpy()
                          > self._alpha_gate_tau).squeeze(-1)
            if self._alpha_gate:
                gated = lab_last.copy()
                gated[~alpha_last] = 0
                self._pending_labels = gated
            else:
                self._pending_labels = lab_last
            self._last_semantic_raw = lab_last
            self._live_alpha = alpha_last
            self.last_coverage = (cov_win if cov_win is not None else
                                  float(target_alpha[0, -1].detach().float().mean().item()))
            t["diffusion"] = 0.0
            t["decode"] = 0.0
            t["total"] = time.perf_counter() - t0
            self.last_timings = t
            K_np = scene["K"][0].detach().cpu().float().numpy()
            w2c_np = w2c[-1].detach().cpu().float().numpy()
            return rgb_frame, K_np, w2c_np

        t1 = time.perf_counter()
        sink: dict = {}
        orig_decode = pipe.vae.decode
        _dual = _make_dual_decode(orig_decode, sink)

        def _probe_decode(latents, **kw):
            # 2026-09-18: record what the VAE was asked to decode, so a missing
            # semantic half can be diagnosed from the log instead of guessed.
            sink.setdefault("latent_shapes", []).append(tuple(latents.shape))
            return _dual(latents, **kw)
        pipe.vae.decode = _probe_decode
        try:
            with torch.no_grad():
                generated = pipe(
                    prompt=self.cfg.prompt,
                    negative_prompt=self.cfg.negative_prompt,
                    seed=0, rand_device=device,
                    height=self.H, width=self.W, num_frames=k,
                    cfg_scale=self.cfg.cfg_scale,
                    num_inference_steps=4 if self.cfg.use_lora else 50,
                    tiled=False,
                    source_views=scene["views"],
                    target_rgb=target_rgb, target_depth=target_depth,
                    target_mask=target_mask,
                    target_poses=c2w.unsqueeze(0),
                    target_intrs=K_rep.unsqueeze(0),
                    target_semantic=target_semantic,
                )
        finally:
            pipe.vae.decode = orig_decode
        t["diffusion"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        head = getattr(pipe, "semantic_class_head", None)
        if "sem_video" in sink:
            labels, _ = _sem_video_to_labels_and_colorized(sink["sem_video"], head=head)
            lab_last = labels[-1].astype(np.int8)
        else:
            # 2026-09-18 (Joana): this used to degrade SILENTLY to the raster
            # hint. The whole trust pipeline ran through it: every "generated
            # label" it graded was the hint (panels pixel-identical, A_map =
            # 1.000 on six pairs). A score that is not the generator's must
            # never pass as one -- fail loudly unless the caller opted in.
            _msg = (f"[LiveDiffusedBackend] no semantic half decoded: VAE latent shapes "
                    f"{sink.get('latent_shapes')} (32 channels expected), sink keys {list(sink)}")
            if not bool(getattr(self, "allow_hint_fallback", False)):
                raise RuntimeError(_msg + " -- set allow_hint_fallback=True to use the raster hint instead")
            print(_msg + " -- USING THE RASTER HINT", flush=True)
            lab_last = target_semantic[0, -1].detach().cpu().numpy().astype(np.int8)
        alpha_last = (target_alpha[0, -1].detach().float().cpu().numpy()
                      > self._alpha_gate_tau).squeeze(-1)
        if self._alpha_gate:
            gated = lab_last.copy()
            gated[~alpha_last] = 0
            self._pending_labels = gated
        else:
            self._pending_labels = lab_last
        self._last_semantic_raw = lab_last
        self._live_alpha = alpha_last
        # 2026-09-04: the view's mean alpha, the SAME statistic the batched
        # training path reports (vec_live_env.last_alpha[i].mean()). Until now
        # the single-env path -- every eval -- reported nothing, so the
        # coherence cost and the incoherent terminal were INERT in evals
        # while live in training (the env printed a WARNING nobody grepped).
        self.last_coverage = (cov_win if cov_win is not None else
                              float(target_alpha[0, -1].detach().float().mean().item()))
        t["decode"] = time.perf_counter() - t2
        t["total"] = time.perf_counter() - t0
        self.last_timings = t

        rgb_frame = np.array(generated[-1])
        K_np = scene["K"][0].detach().cpu().float().numpy()
        w2c_np = w2c[-1].detach().cpu().float().numpy()
        return rgb_frame, K_np, w2c_np

    def _rasterize_labels(self, scene, pose_recon, t_idx: int = 0):
        # Calibrated.render() stores this as _last_semantic_image right after
        # the RGB dispatch; serve the labels DECODED FROM THE SAME DIFFUSION
        # CALL instead of a raster pass, so the reward reads what the policy saw.
        if self._pending_labels is not None:
            return self._pending_labels
        return super()._rasterize_labels(scene, pose_recon, t_idx)
