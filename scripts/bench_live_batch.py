"""Benchmark batched live diffusion: how many robots can one GPU serve?

The parallel-training plan hinges on one number: the per-robot cost of a
BATCHED generation call (N independent 5-frame sequences through the pipe at
once) vs the batch-1 baseline (~1.4 s/step). This script measures it without
building the vectorized env first.

For each batch size B it fabricates B plausible pose histories (spread along
the recorded trajectory on different lanes), rasterizes B*k conditioning
views, stacks the targets on the batch dimension, and times the pipe call.
A batch size the pipeline cannot handle prints its exception and moves on —
that failure locates the exact surgery the batched backend needs.

Usage (GPU node):
    python scripts/bench_live_batch.py \
        --scene rugd_trail_00 --clips_dir ... --poses_dir ... --labels_dir ... \
        --batches 1,2,4,8 --repeats 3
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from src.env.real_calibrated import CalibratedBackendConfig
from src.env.live_backend import LiveDiffusedBackend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--live_ckpt",
                    default="/scratch/m000204-pm07b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors")
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--num_steps", type=int, default=4,
                    help="diffusion sampler steps (distilled default 4; "
                         "2 is the next speed rung, quality-gated by eyes)")
    ap.add_argument("--save_samples", default="",
                    help="dir to save generated rgb+semantic frame strips per "
                         "robot (the quality gate for resolution sweeps)")
    args = ap.parse_args()
    if args.height % 112 or args.width % 112:
        raise SystemExit(
            "height/width must be multiples of 112: the WorldMirror "
            "reconstructor ViT uses 14-px patches and the VAE(8x)+DiT(patch 2) "
            "chain needs 16 — lcm=112. Valid: 112/224/336/448/560.")

    import torch
    from diffsynth.utils.auxiliary import homo_matrix_inverse
    from inference_semantic import (
        _decoded_video_to_uint8, _make_dual_decode,
        _sem_video_to_labels_and_colorized,
    )

    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=args.height,
        W=args.width,
    )
    world = LiveDiffusedBackend(cfg, checkpoint=args.live_ckpt)
    world.load_scene(args.scene)
    scene = world._cache[args.scene]
    pipe = world._pipe
    device = next(pipe.reconstructor.parameters()).device
    H, W, k = world.H, world.W, args.frames

    # B robots: histories sampled at different trajectory arcs + lateral lanes.
    src_c2w = scene["cam2world"].detach().cpu().numpy()          # [N,4,4]
    n_src = len(src_c2w)

    def history(robot_i: int, B: int) -> np.ndarray:
        base = int((robot_i + 1) * n_src / (B + 2))
        hist = []
        for j in range(k):
            m = src_c2w[min(base + j, n_src - 1)].copy()
            m[:3, 3] += m[:3, 1] * (0.4 * (robot_i % 3 - 1))     # lane offset
            hist.append(m)
        return np.stack(hist)                                     # [k,4,4]

    raster = pipe.reconstructor.gs_renderer.rasterizer
    src_pos = scene["cam2world"][:, :3, 3]

    for B in [int(x) for x in args.batches.split(",")]:
        try:
            c2w = torch.from_numpy(np.stack([history(i, B) for i in range(B)])).to(
                device=device, dtype=scene["cam2world"].dtype)    # [B,k,4,4]
            w2c = homo_matrix_inverse(c2w.reshape(B * k, 4, 4))
            K_rep = scene["K"][0:1].repeat(B * k, 1, 1)
            d = torch.cdist(c2w.reshape(B * k, 4, 4)[:, :3, 3].float(), src_pos.float())
            target_ts = scene["timestamps"][d.argmin(dim=1)]

            rgb, depth, alpha = raster.forward(
                scene["gaussians"], render_viewmats=[w2c], render_Ks=[K_rep],
                render_timestamps=[target_ts], sh_degree=0, width=W, height=H)
            sem, _, _ = raster.forward(
                scene["gaussians"], render_viewmats=[w2c], render_Ks=[K_rep],
                render_timestamps=[target_ts], sh_degree=0, width=W, height=H,
                feature="labels")

            def bshape(x):   # [1, B*k, ...] -> [B, k, ...]
                return x.reshape(B, k, *x.shape[2:]) if x.shape[0] == 1 else x

            tgt_rgb = bshape(rgb)
            tgt_depth = bshape(depth)
            tgt_mask = (bshape(alpha) > 1.0).float()
            tgt_sem = bshape(sem.argmax(dim=-1).to(torch.long))

            times = []
            for r in range(args.repeats):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                sink = {}
                orig_decode = pipe.vae.decode
                pipe.vae.decode = _make_dual_decode(orig_decode, sink)
                with torch.no_grad():
                    pipe(prompt=[cfg.prompt] * B,
                         negative_prompt=[cfg.negative_prompt] * B,
                         seed=r, rand_device=device, height=H, width=W,
                         num_frames=k, cfg_scale=cfg.cfg_scale,
                         num_inference_steps=args.num_steps, tiled=False,
                         source_views=scene["views"],
                         target_rgb=tgt_rgb, target_depth=tgt_depth,
                         target_mask=tgt_mask,
                         target_poses=c2w, target_intrs=K_rep.reshape(B, k, 3, 3),
                         target_semantic=tgt_sem)
                pipe.vae.decode = orig_decode
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            vram = torch.cuda.max_memory_allocated() / 2**30
            t_med = float(np.median(times))
            print(f"B={B}: {t_med:.2f} s/call = {t_med / B:.2f} s/robot-step  "
                  f"peak VRAM {vram:.1f} GB  (times: {[f'{x:.2f}' for x in times]})",
                  flush=True)
            if args.save_samples:
                from PIL import Image
                out_dir = Path(args.save_samples) / f"{args.width}x{args.height}_B{B}"
                out_dir.mkdir(parents=True, exist_ok=True)
                rv = sink.get("rgb_video")
                sv = sink.get("sem_video")
                head = getattr(pipe, "semantic_class_head", None)
                alpha_b = bshape(alpha)
                for i in range(min(B, 2)):
                    if rv is not None and getattr(rv, "ndim", 0) == 5:
                        frames = _decoded_video_to_uint8(rv[i:i + 1])
                        Image.fromarray(np.concatenate(list(frames), axis=1)).save(
                            out_dir / f"robot{i}_rgb.png")
                    if sv is not None and getattr(sv, "ndim", 0) == 5:
                        _, sem_rgb = _sem_video_to_labels_and_colorized(sv[i], head=head)
                        Image.fromarray(np.concatenate(list(sem_rgb), axis=1)).save(
                            out_dir / f"robot{i}_semantic.png")
                    # ground truth the model was given: rasterized conditioning
                    # (the NON-hallucinated reference) + which pixels were known
                    rast = (tgt_rgb[i].detach().float().cpu().numpy()
                            * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(np.concatenate(list(rast), axis=1)).save(
                        out_dir / f"robot{i}_raster.png")
                    known = (alpha_b[i].detach().float().cpu().numpy() > 0.5)
                    known = (known[..., 0] if known.ndim == 4 else known)
                    known = (known * 255).astype(np.uint8)
                    Image.fromarray(np.concatenate(list(known), axis=1)).save(
                        out_dir / f"robot{i}_known_mask.png")
                print(f"    samples -> {out_dir}", flush=True)
        except Exception:
            print(f"B={B}: FAILED —")
            traceback.print_exc()
            print("(this traceback locates the batch-support surgery needed)", flush=True)


if __name__ == "__main__":
    main()
