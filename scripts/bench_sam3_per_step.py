"""Cost per step: co-generated semantics vs SAM3 run on the generated frame.

The question a reviewer will ask about reading the reward from the world
model's own semantic channel: why not segment the generated RGB with an
off-the-shelf model at every step? The defensible answer is cost, so measure
it. Two processes, because SAM3 lives in the `sam3` conda env and the world
model in `neoverse` (and the pipeline leaves ~2 GB of VRAM, which SAM3 would
not fit beside anyway):

  --mode render  (neoverse python): walk the recorded trail exactly as
                 live_benchmark.py does, time every live render (one diffusion
                 pass -> RGB AND semantics), save each generated frame as PNG.
  --mode sam3    (sam3 python): load those PNGs, time SAM3 with the labelling
                 pipeline's own loop (vision encoded once, one text prompt per
                 class, sam3_precompute_labels.py), print the table + JSON.

Single-frame numbers on both sides, same GPU type. Training batches four envs
per diffusion call and SAM3 could batch too, so the RATIO is the portable
number, not either absolute. Launch: scripts/slurm/bench_sam3.sh
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
for _p in (REPO_ROOT.parent / "NeoVerse", Path("/scratch/m000204-pm07b/joana/NeoVerse")):
    if _p.exists():
        sys.path.insert(0, str(_p))


def run_render(args) -> None:
    from src.env.real_calibrated import CalibratedBackendConfig
    from src.env.live_backend import LiveDiffusedBackend
    from live_benchmark import walk_poses, resized_labels
    import cv2

    clips_dir = args.clips_dir or (
        "/scratch/m000204-pm07b/joana/data/gnd_clips" if args.scene.startswith("gnd_")
        else "/scratch/m000204-pm07b/joana/data/rugd_clips")
    frames_dir = args.out / f"frames_{args.scene}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    labels_path = resized_labels(Path(args.labels_dir) / (args.scene + ".npz"),
                                 (args.width, args.height), args.out)
    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: str(labels_path)},
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
    )
    cfg.W, cfg.H = args.width, args.height
    cfg.sem_palette_version = args.palette
    backend = LiveDiffusedBackend(cfg, checkpoint=args.checkpoint, live_frames=args.frames)
    backend.load_scene(args.scene)
    cal = backend._calib[args.scene]
    poses = walk_poses(cal, args.steps + 2)

    rows = []
    backend._pose_hist = []
    for i, P in enumerate(poses):
        rgb, _, _ = backend.render(P)
        if i < 2:                      # warmup
            continue
        t = dict(backend.last_timings)
        rows.append(dict(render=t["total"], raster=t["raster"],
                         diffusion=t["diffusion"], decode=t["decode"]))
        cv2.imwrite(str(frames_dir / f"step_{i - 2:03d}.png"),
                    np.ascontiguousarray(rgb[:, :, ::-1]))
        print(f"  step {i - 1}/{args.steps}: render {t['total']:.2f}s", flush=True)

    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out = dict(scene=args.scene, resolution=f"{args.width}x{args.height}",
               frames_per_call=args.frames, steps=len(rows), means=m,
               frames_dir=str(frames_dir))
    (args.out / f"render_{args.scene}.json").write_text(json.dumps(out, indent=2))
    print(f"[render] {m['render']:.2f} s/step over {len(rows)} steps "
          f"(raster {m['raster']:.2f} diffusion {m['diffusion']:.2f} decode {m['decode']:.2f})")
    print(f"==> {len(rows)} frames in {frames_dir}")


def run_sam3(args) -> None:
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    from sam3_precompute_labels import CLASSES

    render = json.loads((args.out / f"render_{args.scene}.json").read_text())
    frames = sorted(Path(render["frames_dir"]).glob("step_*.png"))
    if not frames:
        raise SystemExit(f"no frames under {render['frames_dir']} -- run --mode render first")
    names = [n for n, _c, _t, _p in CLASSES]
    print(f"loading {args.sam3_model} ({len(names)} class prompts) for {len(frames)} frames ...",
          flush=True)
    sam = Sam3Model.from_pretrained(args.sam3_model, dtype=torch.bfloat16, device_map="auto")
    proc = Sam3Processor.from_pretrained(args.sam3_model)
    sam.eval()

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    rows = []
    for i, fp in enumerate(frames):
        img = Image.open(fp).convert("RGB")
        sync()
        t0 = time.perf_counter()
        img_inputs = proc(images=img, return_tensors="pt").to(sam.device)
        with torch.no_grad():
            vis = sam.get_vision_features(pixel_values=img_inputs.pixel_values)
        sync()
        t_enc = time.perf_counter() - t0
        target_sizes = img_inputs.get("original_sizes").tolist()
        t1 = time.perf_counter()
        for name in names:
            text_inputs = proc(text=name, return_tensors="pt").to(sam.device)
            with torch.no_grad():
                out = sam(vision_embeds=vis, **text_inputs)
            proc.post_process_instance_segmentation(
                out, threshold=args.conf, mask_threshold=args.mask_threshold,
                target_sizes=target_sizes)
        sync()
        t_prompts = time.perf_counter() - t1
        if i < 2:                      # warmup
            continue
        rows.append(dict(sam3_encode=t_enc, sam3_prompts=t_prompts, sam3_total=t_enc + t_prompts))
        print(f"  frame {i - 1}/{len(frames) - 2}: sam3 {t_enc + t_prompts:.2f}s", flush=True)

    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    r = render["means"]
    n_prompts = len(names)
    per_prompt = m["sam3_prompts"] / n_prompts
    sam3_14 = m["sam3_encode"] + 14 * per_prompt
    summary = dict(scene=args.scene, resolution=render["resolution"],
                   frames_per_call=render["frames_per_call"], render_steps=render["steps"],
                   sam3_frames=len(rows), n_prompts=n_prompts, render=r, sam3=m,
                   sam3_per_prompt=per_prompt, sam3_14_prompts=sam3_14,
                   overhead_29=m["sam3_total"] / r["render"],
                   overhead_14=sam3_14 / r["render"])
    (args.out / f"sam3_bench_{args.scene}.json").write_text(json.dumps(summary, indent=2))

    line = "=" * 72
    print(f"\n{line}\nCOST PER STEP, single frame, {render['resolution']}, {args.scene}\n{line}")
    print(f"{'world model, one pass (RGB + semantics)':<44}{r['render']:7.2f} s/step   "
          f"raster {r['raster']:.2f}  diffusion {r['diffusion']:.2f}  decode {r['decode']:.2f}")
    print(f"{'SAM3 on the generated frame, ' + str(n_prompts) + ' prompts':<44}"
          f"{m['sam3_total']:7.2f} s/frame  encode {m['sam3_encode']:.2f}  "
          f"prompts {m['sam3_prompts']:.2f}  ({per_prompt * 1000:.0f} ms each)")
    print(f"{'SAM3 with 14 prompts (extrapolated)':<44}{sam3_14:7.2f} s/frame")
    print(f"{'RGB pass + SAM3 (' + str(n_prompts) + ')':<44}{r['render'] + m['sam3_total']:7.2f} s/step   "
          f"= {100 * summary['overhead_29']:.0f}% more than co-generation")
    print(f"{'RGB pass + SAM3 (14)':<44}{r['render'] + sam3_14:7.2f} s/step   "
          f"= {100 * summary['overhead_14']:.0f}% more than co-generation")
    print(f"\nover a 200k-step run that is {200_000 * m['sam3_total'] / 3600:.0f} h extra "
          f"({n_prompts} prompts) or {200_000 * sam3_14 / 3600:.0f} h extra (14 prompts), "
          f"before any batching on either side.")
    print(f"==> {args.out / ('sam3_bench_' + args.scene + '.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("render", "sam3"), required=True)
    ap.add_argument("--scene", default="gnd_AUw360")
    ap.add_argument("--out", type=Path,
                    default=Path("/scratch/m000204-pm07b/joana/outputs/sam3_bench"))
    # render mode
    ap.add_argument("--checkpoint", default=None, help="semantics LoRA the policy trained on")
    ap.add_argument("--clips_dir", default=None,
                    help="default: gnd_clips for gnd_* scenes, rugd_clips otherwise")
    ap.add_argument("--poses_dir", default="/scratch/m000204-pm07b/joana/outputs/poses")
    ap.add_argument("--labels_dir",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14")
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--palette", type=int, default=4)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--frames", type=int, default=5, help="frames per live call (training: 5)")
    ap.add_argument("--steps", type=int, default=20)
    # sam3 mode
    ap.add_argument("--sam3_model", default="facebook/sam3")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "render":
        if not args.checkpoint:
            raise SystemExit("--mode render needs --checkpoint")
        run_render(args)
    else:
        run_sam3(args)


if __name__ == "__main__":
    main()
