"""Batched live-diffusion training: N robots, one pipe, one generation per step.

Why: live training at batch 1 costs ~1.4 s/step — 48 h buys ~120k steps, the
budget pure-live runs kept dying at. The GPU is mostly idle at batch 1, so N
independent robots batched into ONE diffusion call amortize the cost to a
fraction per robot. SB3's own vectorization cannot do this: SubprocVecEnv
duplicates the 40 GB pipe per process, DummyVecEnv steps envs sequentially
(N separate batch-1 calls). This module centralizes the render:

    LiveVecEnv.step_wait():
        1. every SceneEnv advances its sim state CHEAPLY (defer_render=True)
        2. all robots' new poses -> BatchedLiveDiffusedBackend.render_batch()
           -> ONE pipe call with batch dim B
        3. frames + generated labels injected back into each env
        4. done envs get a correct terminal observation, then reset (their
           spawn views render in a second, usually tiny, batched call)

All robots must live in the SAME scene (one gaussian field on the GPU); the
scene rotates per training run, not per episode.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .live_backend import LiveDiffusedBackend


class InjectedLabelBackend:
    """SemanticBackend shim for batched-live envs: labels arrive via
    SceneEnv.inject_render() instead of a backend-side render."""

    def __init__(self, env):
        self._env = env

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        labels = getattr(self._env, "_injected_labels", None)
        if labels is None:
            raise RuntimeError("InjectedLabelBackend: no labels injected yet")
        return labels


class BatchedLiveDiffusedBackend(LiveDiffusedBackend):
    """LiveDiffusedBackend with per-robot pose histories and a batched render.

    render() (the single-robot path) still works — evals and smokes reuse it.
    render_batch() serves the vectorized env.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._hists: dict[int, list] = {}
        self.hist_jump_m = 0.5      # a pose jump beyond this drops the history (reset)

    def _robot_hist(self, robot_id: int, pose_recon: np.ndarray) -> list:
        from .live_backend import cold_history, respace_history, INTERP_PATH_LEN
        hist = self._hists.get(robot_id, [])
        if hist:
            # 2026-09-06: the jump was compared in RECON UNITS against 0.5, but
            # the reconstruction is normalized (scale_m_per_unit metres per
            # unit), so a spawn several metres away did not count as a jump
            # and the new episode's first frame was conditioned on the
            # previous episode's poses (the washed-out step-0 frames Joana
            # flagged). Compare in metres, and resets clear explicitly.
            _cal = self._calib.get(self._current_scene_id)
            _m_per_unit = float(getattr(_cal, "scale", 1.0) or 1.0)
            jump_m = float(np.linalg.norm(hist[-1][:3, 3] - pose_recon[:3, 3])) * _m_per_unit
            if jump_m > float(getattr(self, "hist_jump_m", 0.5)):
                hist = []
                self.hist_resets_jump = int(getattr(self, "hist_resets_jump", 0)) + 1
        if not hist:
            scene = self._cache.get(self._current_scene_id)
            hist = cold_history(pose_recon, scene, self.live_frames)
        else:
            _keep = INTERP_PATH_LEN if getattr(self, "interp_history", False) else self.live_frames
            hist = (hist + [pose_recon])[-_keep:]
        self._hists[robot_id] = hist
        if getattr(self, "interp_history", False):
            return respace_history(hist, self._cache.get(self._current_scene_id), self.live_frames)
        return hist

    def render_batch(self, items: "list[tuple[int, np.ndarray]]"):
        """items: [(robot_id, pose_nav_robot 4x4), ...] -> per item
        (rgb HxWx3 uint8, K 3x3, w2c_nav 4x4, labels HxW int8)."""
        import torch
        from diffsynth.utils.auxiliary import homo_matrix_inverse
        from inference_semantic import (
            _decoded_video_to_uint8, _make_dual_decode,
            _sem_video_to_labels_and_colorized,
        )

        scene = self._cache.get(self._current_scene_id)
        if scene is None:
            raise RuntimeError("call load_scene() first")
        cal = self._calib[self._current_scene_id]
        pipe = self._pipe
        recon = self._recon()
        device = next(recon.parameters()).device
        # 2026-08-30: the startup pre-reconstruction pass parks scene caches on
        # CPU; whatever path re-activates the resident scene can miss the
        # upload (cdist device crash, G12 456804). Guard here — the render is
        # the one place that MUST have the scene on the GPU.
        if torch.is_tensor(scene.get("K")) and scene["K"].device.type == "cpu":
            from .real_backend import _move_tree_to
            scene = _move_tree_to(scene, device)
            self._cache[self._current_scene_id] = scene
        B, k = len(items), self.live_frames
        t0 = time.perf_counter()

        hists = []
        for robot_id, pose_nav in items:
            pose_recon, _ = self._pose_nav_to_recon(pose_nav)
            hists.append(self._robot_hist(robot_id, pose_recon.astype(np.float32)))
        # HISTORY SPACING (2026-09-16): the 09-07 note claimed the chunked arm's
        # generator clip had poses 2.5 m apart. Print the median spacing of the
        # first robot's clip every 200 renders so the claim is settled by a number
        # (sub-step spacing ~0.25 m = no gap).
        self._hist_print_n = int(getattr(self, "_hist_print_n", 0)) + 1
        if self._hist_print_n % 200 == 1 and len(hists[0]) >= 2:
            _cal = self._calib.get(self._current_scene_id)
            _mpu = float(getattr(_cal, "scale", 1.0) or 1.0)
            _p = np.stack([h[:3, 3] for h in hists[0]])
            _d = np.linalg.norm(np.diff(_p, axis=0), axis=1) * _mpu
            from .live_backend import _rot_angle
            _turn = [np.degrees(_rot_angle(hists[0][i][:3, :3].astype(np.float64), hists[0][i + 1][:3, :3].astype(np.float64)))
                     for i in range(len(hists[0]) - 1)]
            print(f"[VecLiveEnv] generator clip turn per frame: max {max(_turn):.1f} deg "
                  f"(interp_history={bool(getattr(self, 'interp_history', False))})", flush=True)
            print(f"[VecLiveEnv] generator clip of robot {items[0][0]}: {len(hists[0])} poses, "
                  f"spacing median {float(np.median(_d)):.2f} m (min {float(_d.min()):.2f}, max {float(_d.max()):.2f}) "
                  f"[render {self._hist_print_n}]", flush=True)

        c2w = torch.from_numpy(np.stack([np.stack(h) for h in hists])).to(
            device=device, dtype=scene["cam2world"].dtype)        # [B,k,4,4]
        flat = c2w.reshape(B * k, 4, 4)
        w2c = homo_matrix_inverse(flat)
        K_rep = scene["K"][0:1].repeat(B * k, 1, 1)
        src_pos = scene["cam2world"][:, :3, 3]
        t_idx = torch.cdist(flat[:, :3, 3].float(), src_pos.float()).argmin(dim=1)
        target_ts = scene["timestamps"][t_idx]
        # which SOURCE FRAME's time each view is rendered at (the scene is
        # time-associated: when the nearest source camera changes, the set of
        # Gaussians rendered changes with it). Exposed for the replay log.
        try:
            self.last_t_idx = t_idx.detach().cpu().numpy().reshape(B, k)[:, -1].tolist()
        except Exception:
            self.last_t_idx = None

        raster = recon.gs_renderer.rasterizer
        t_last = t_idx.reshape(B, k)[:, -1].tolist()
        _rw = int(getattr(self.cfg, "render_window", 0) or 0)
        if _rw > 0:
            # one Gaussian window per robot -> B batches of k views each
            from .window import window_gaussians
            _gs = [window_gaussians(scene["gaussians"][0], int(t_last[b]), _rw) for b in range(B)]
            _vm = [w2c.reshape(B, k, 4, 4)[b] for b in range(B)]
            _ks = [K_rep.reshape(B, k, 3, 3)[b] for b in range(B)]
            _ts = [target_ts.reshape(B, k)[b] for b in range(B)]
        else:
            _gs, _vm, _ks, _ts = scene["gaussians"], [w2c], [K_rep], [target_ts]
        rgb_t, depth_t, alpha_t = raster.forward(
            _gs, render_viewmats=_vm, render_Ks=_ks,
            render_timestamps=_ts, sh_degree=0,
            width=self.W, height=self.H)
        # coverage window: opacity of the w nearest frames' Gaussians at each
        # robot's LAST view; consumed by _render_and_inject instead of last_alpha
        _cw = int(getattr(self.cfg, "coverage_window", 0) or 0)
        self.last_alpha_window = None
        if _cw > 0:
            from .window import windowed_alpha
            _aw = []
            for b in range(B):
                a = windowed_alpha(raster, scene["gaussians"][0],
                                   w2c.reshape(B, k, 4, 4)[b, -1], K_rep.reshape(B, k, 3, 3)[b, -1],
                                   target_ts.reshape(B, k)[b, -1], int(t_last[b]), _cw,
                                   self.W, self.H)
                _aw.append(None if a is None else a.detach().float().cpu().numpy().reshape(self.H, self.W))
            self.last_alpha_window = _aw
        # Coverage telemetry (2026-08-31, her spawn-certification ask): mean
        # reconstruction alpha of the LAST frame per item = how much of the
        # view is backed by real Gaussians vs inpainted. Read-only attr;
        # consumed by drive_preview's HUD.
        try:
            # Shape-robust: whatever the leading layout, total elements are
            # (B*k*H*W[*1]) in flat stacking order — reshape and take the
            # LAST history frame per item.
            self.last_coverage = float(
                alpha_t.reshape(B, k, -1)[:, -1].float().mean())
        except Exception as e:
            print(f"[coverage] failed on alpha shape "
                  f"{tuple(alpha_t.shape)}: {e}", flush=True)
            self.last_coverage = float("nan")
        sem_t, _, _ = raster.forward(
            _gs, render_viewmats=_vm, render_Ks=_ks,
            render_timestamps=_ts, sh_degree=0,
            width=self.W, height=self.H, feature="labels")

        def bshape(x):
            return x.reshape(B, k, *x.shape[2:]) if x.shape[0] == 1 else x

        tgt_rgb, tgt_depth = bshape(rgb_t), bshape(depth_t)
        tgt_alpha = bshape(alpha_t)
        tgt_mask = (tgt_alpha > 1.0).float()
        tgt_sem = bshape(sem_t.argmax(dim=-1).to(torch.long))

        # Splat-raster debug view (2026-08-31, her ask): the raw Gaussian
        # raster the diffusion is conditioned on, LAST history frame per item.
        # Read-only attr, consumed by drive_preview's RASTER panel — shows
        # where the splats end and diffusion hallucination begins.
        try:
            arr = tgt_rgb.reshape(B, k, -1)[:, -1].detach().float().cpu().numpy()
            arr = arr.reshape(B, self.H, self.W, -1)[..., :3]
            if arr.max() <= 1.5:
                arr = arr * 255.0
            self.last_raster = [np.clip(arr[i], 0, 255).astype(np.uint8)
                                for i in range(B)]
        except Exception as e:
            print(f"[raster] debug view failed on shape "
                  f"{tuple(tgt_rgb.shape)}: {e}", flush=True)
            self.last_raster = None

        # Gaussian-label raster (her ask 2026-09-01): the semantic labels
        # carried by the SPLATS themselves, before the diffusion rewrites
        # them — the honest "what does the reconstruction actually know"
        # semantic view, as opposed to the generated one.
        try:
            _sr = (tgt_sem.reshape(B, k, -1)[:, -1].detach().cpu().numpy()
                   .reshape(B, self.H, self.W))
            self.last_sem_raster = [_sr[i].astype(np.int16) for i in range(B)]
        except Exception as e:
            print(f"[raster] sem view failed on shape "
                  f"{tuple(tgt_sem.shape)}: {e}", flush=True)
            self.last_sem_raster = None

        # Per-pixel SUPPORT map (2026-09-01, her ask): the raw alpha of the
        # last history frame per item. This is the quantity the gate
        # thresholds and the one the `cov` summary averages — exposing the map
        # itself lets us SEE where the world model has evidence and where it
        # is inventing, instead of arguing about a single number.
        try:
            _a = (tgt_alpha.reshape(B, k, -1)[:, -1].detach().float()
                  .cpu().numpy().reshape(B, self.H, self.W))
            self.last_alpha = [_a[i] for i in range(B)]
        except Exception as e:
            print(f"[raster] alpha view failed on shape "
                  f"{tuple(tgt_alpha.shape)}: {e}", flush=True)
            self.last_alpha = None

        if self.raster_obs:
            # Raster-observation ablation: serve the splat raster and the
            # splat labels computed above; the pipe is never called.
            if self.last_raster is None or self.last_sem_raster is None:
                raise RuntimeError("raster_obs: the raster views failed, no observation to serve")
            rgbs = [np.ascontiguousarray(r[..., :3]) for r in self.last_raster]
            labels_per_robot = [sr.astype(np.int8) for sr in self.last_sem_raster]
        else:
            sink: dict = {}
            orig_decode = pipe.vae.decode
            pipe.vae.decode = _make_dual_decode(orig_decode, sink)
            try:
                with torch.no_grad():
                    generated = pipe(
                        prompt=[self.cfg.prompt] * B,
                        negative_prompt=[self.cfg.negative_prompt] * B,
                        seed=0, rand_device=device,
                        height=self.H, width=self.W, num_frames=k,
                        cfg_scale=self.cfg.cfg_scale,
                        num_inference_steps=getattr(
                            self, "num_inference_steps",
                            4 if self.cfg.use_lora else 50),
                        tiled=False,
                        source_views=scene["views"],
                        target_rgb=tgt_rgb, target_depth=tgt_depth,
                        target_mask=tgt_mask,
                        target_poses=c2w,
                        target_intrs=K_rep.reshape(B, k, 3, 3),
                        target_semantic=tgt_sem,
                    )
            finally:
                pipe.vae.decode = orig_decode

            # ---- unpack last frames per robot ----
            # The pipe's own return is USELESS for B>1: vae_output_to_video does
            # reduce("B C T H W -> T H W C", "mean") — it AVERAGES the robots into
            # one ghost video. The dual-decode sink captures the raw VAE decode
            # BEFORE that reduce, batch dim intact — per-robot frames come from it.
            def last_rgb_frames() -> list:
                rv = sink.get("rgb_video")
                if rv is not None and getattr(rv, "ndim", 0) == 5 and rv.shape[0] == B:
                    return [_decoded_video_to_uint8(rv[i:i + 1])[-1] for i in range(B)]
                if B == 1:
                    return [np.array(generated[-1])]
                raise RuntimeError(
                    f"render_batch: no batched rgb_video in dual-decode sink for "
                    f"B={B} (keys={list(sink)}); the pipe return is batch-averaged "
                    "and cannot be unpacked per robot")

            rgbs = last_rgb_frames()

            head = getattr(pipe, "semantic_class_head", None)
            labels_per_robot: list = []
            sem_video = sink.get("sem_video")
            if sem_video is not None:
                sv = sem_video
                if B == 1:
                    lab, _ = _sem_video_to_labels_and_colorized(sv, head=head)
                    labels_per_robot = [lab[-1].astype(np.int8)]
                elif hasattr(sv, "shape") and sv.shape[0] == B:
                    for i in range(B):
                        lab, _ = _sem_video_to_labels_and_colorized(sv[i], head=head)
                        labels_per_robot.append(lab[-1].astype(np.int8))
            if not labels_per_robot:
                # 2026-09-18 (Joana): was a SILENT degrade to the raster hint;
                # the single-frame path's version of this fed the whole trust
                # pipeline hint labels as "generated". Fail loudly unless the
                # caller opted in -- a reward that is not the generator's must
                # never pass as one.
                _sv_shape = getattr(sem_video, "shape", None)
                _msg = (f"[BatchedLiveDiffusedBackend] no semantic half decoded for B={B}: "
                        f"sink keys {list(sink)}, sem_video shape {_sv_shape}")
                if not bool(getattr(self, "allow_hint_fallback", False)):
                    raise RuntimeError(_msg + " -- set allow_hint_fallback=True to use the raster hint")
                print(_msg + " -- USING THE RASTER HINT", flush=True)
                labels_per_robot = [
                    tgt_sem[i, -1].detach().cpu().numpy().astype(np.int8)
                    for i in range(B)]


        alpha_last = (tgt_alpha[:, -1].detach().float().cpu().numpy()
                      > getattr(self, "_alpha_gate_tau", 0.5))
        if alpha_last.ndim == 4:
            alpha_last = alpha_last.squeeze(-1)

        K_np = scene["K"][0].detach().cpu().float().numpy()
        nav2recon = cal.nav_world_to_recon_homogeneous()
        out = []
        for i in range(B):
            lab = labels_per_robot[i]
            if self._alpha_gate:
                lab = lab.copy()
                lab[~alpha_last[i]] = 0
            w2c_recon = homo_matrix_inverse(
                c2w[i, -1:].detach()).squeeze(0).cpu().float().numpy()
            w2c_nav = (w2c_recon.astype(np.float64) @ nav2recon).astype(np.float32)
            out.append((rgbs[i], K_np, w2c_nav, lab))
        self.last_timings = {"total": time.perf_counter() - t0, "batch": B}
        return out


try:
    from stable_baselines3.common.vec_env import VecEnv
except Exception:                                     # import-time safety only
    VecEnv = object


class LiveVecEnv(VecEnv):
    """N SceneEnvs (defer_render) sharing one BatchedLiveDiffusedBackend.

    Multi-scene (2026-08-29, Joana's call: "the policy must learn what
    obstacles look like — one world can't teach that"): pass `scenes` and
    `rotate_every` and the env rotates the resident world on a step budget —
    all episodes force-truncate cleanly, the next scene's Gaussians load,
    pose histories clear (cold-start walk-ins rebuild), robots respawn there.
    """

    def __init__(self, envs: list, backend: BatchedLiveDiffusedBackend,
                 scenes: "list[str] | None" = None, rotate_every: int = 0,
                 start_idx: int = 0):
        self.envs = envs
        self.backend = backend
        self.scenes = list(scenes) if scenes else []
        self.rotate_every = int(rotate_every)
        # 2026-09-23: every GPU worker used to start at scene 0 and rotate on its
        # own clock, so all K workers sat on the SAME scene at the same time and a
        # 12 h 4-GPU run only ever reached the first six of thirteen scenes (job
        # 499454: 5 rotations per worker, never past sequoia1_21). Staggering the
        # start spreads the workers across the list, so every batch mixes scenes
        # and the union covers the whole list in a quarter of the steps.
        self._scene_i = (int(start_idx) % len(self.scenes)) if self.scenes else 0
        self._steps_since_rot = 0
        self._actions: Optional[np.ndarray] = None
        self.render_mode = None
        super().__init__(len(envs), envs[0].observation_space, envs[0].action_space)

    # ---------- helpers ----------

    def _render_and_inject(self, idxs: list) -> None:
        if not idxs:
            return
        items = [(i, self.envs[i]._robot_pose_world) for i in idxs]
        results = self.backend.render_batch(items)
        # last_alpha is indexed by POSITION IN THE BATCH, not by env id — the
        # two coincide only when every env renders. Zip over both together.
        alphas = getattr(self.backend, "last_alpha", None)
        alphas_w = getattr(self.backend, "last_alpha_window", None)   # coverage window
        rasters = getattr(self.backend, "last_raster", None)         # the Gaussian render the generator was conditioned on
        for j, (i, (rgb, K, w2c, lab)) in enumerate(zip(idxs, results)):
            cov = None
            if alphas_w is not None and j < len(alphas_w) and alphas_w[j] is not None:
                cov = float(np.asarray(alphas_w[j]).mean())
            elif alphas is not None and j < len(alphas):
                cov = float(np.asarray(alphas[j]).mean())
            ras = rasters[j] if (rasters is not None and j < len(rasters)) else None
            alp = alphas[j] if (alphas is not None and j < len(alphas)) else None
            self.envs[i].inject_render(rgb, K, w2c, labels=lab, coverage=cov, raster=ras, alpha=alp)

    def _stack_obs(self, obs_list: list) -> dict:
        return {key: np.stack([o[key] for o in obs_list])
                for key in obs_list[0]}

    # ---------- VecEnv API ----------

    def clear_history(self, idxs=None) -> None:
        """Drop the frame history of these robots (all if None) so their next
        render starts a fresh walk-in. Called at every episode reset."""
        if idxs is None:
            self.backend._hists.clear()
        else:
            for i in idxs:
                self.backend._hists.pop(i, None)
        n = int(getattr(self.backend, "hist_resets_explicit", 0)) + (self.num_envs if idxs is None else len(idxs))
        self.backend.hist_resets_explicit = n
        if n in (1, 100, 1000):
            print(f"[VecLiveEnv] frame history cleared at reset: {n} times so far "
                  f"(jump-rule resets {int(getattr(self.backend, 'hist_resets_jump', 0))})", flush=True)

    def reset(self):
        for e in self.envs:
            e.reset()
        self.clear_history()
        self._render_and_inject(list(range(self.num_envs)))
        return self._stack_obs([e._obs() for e in self.envs])

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list = [None] * self.num_envs
        for i, e in enumerate(self.envs):
            _, r, term, trunc, info = e.step(self._actions[i])
            rewards[i] = r
            dones[i] = bool(term or trunc)
            info = dict(info)
            info["TimeLimit.truncated"] = bool(trunc and not term)
            # Which scene produced this step. Nothing logged it, so a reward
            # sawtooth caused by rotation was indistinguishable from the policy
            # getting worse, and the ~10% of episodes force-truncated at a
            # rotation boundary were invisible (2026-09-02).
            info["scene_idx"] = float(self._scene_i)
            infos[i] = info
        self._steps_since_rot += self.num_envs
        rotate = (self.rotate_every > 0 and len(self.scenes) > 1
                  and self._steps_since_rot >= self.rotate_every)
        if rotate:
            print(f"[VecLiveEnv] scene rotate -> "
                  f"{self.scenes[(self._scene_i + 1) % len(self.scenes)]} "
                  f"at {self._steps_since_rot} steps; force-truncating "
                  f"{sum(1 for d in dones if not d)}/{self.num_envs} episodes",
                  flush=True)
            for i in range(self.num_envs):     # force-truncate survivors
                if not dones[i]:
                    dones[i] = True
                    infos[i]["TimeLimit.truncated"] = True
                    # flagged so eval/analysis can drop them: these ended
                    # because the scene changed, not because of the policy
                    infos[i]["rotation_truncated"] = True
        # one batched render at the post-step poses (correct terminal obs too,
        # still in the OLD scene when rotating)
        self._render_and_inject(list(range(self.num_envs)))
        reset_idx = [i for i in range(self.num_envs) if dones[i]]
        for i in reset_idx:
            infos[i]["terminal_observation"] = self.envs[i]._obs()
        if rotate:
            self._scene_i = (self._scene_i + 1) % len(self.scenes)
            new_scene = self.scenes[self._scene_i]
            self.backend._hists.clear()        # cross-scene coords may alias;
            for e in self.envs:                # walk-ins rebuild on reset
                e.scene_ids = [new_scene]
            self._steps_since_rot = 0
        for i in reset_idx:
            self.envs[i].reset()               # load_scene(new) happens here
        self.clear_history(reset_idx)          # a new episode never inherits frames
        self._render_and_inject(reset_idx)     # spawn views, usually tiny
        obs = self._stack_obs([e._obs() for e in self.envs])
        return obs, rewards, dones, infos

    def close(self) -> None:
        for e in self.envs:
            e.close()

    def get_attr(self, attr_name, indices=None):
        idx = range(self.num_envs) if indices is None else np.atleast_1d(indices)
        return [getattr(self.envs[i], attr_name) for i in idx]

    def set_attr(self, attr_name, value, indices=None):
        idx = range(self.num_envs) if indices is None else np.atleast_1d(indices)
        for i in idx:
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        idx = range(self.num_envs) if indices is None else np.atleast_1d(indices)
        return [getattr(self.envs[i], method_name)(*args, **kwargs) for i in idx]

    def env_is_wrapped(self, wrapper_class, indices=None):
        idx = range(self.num_envs) if indices is None else np.atleast_1d(indices)
        return [False for _ in idx]

    def seed(self, seed=None):
        return [None for _ in range(self.num_envs)]
