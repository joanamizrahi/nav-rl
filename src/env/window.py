"""Fusion window: which recorded frames' Gaussians a query may use.

The reconstructor makes one Gaussian per pixel per frame. In static mode they
are all fused into one constant set (timestamp -1), in dynamic mode each frame
keeps its own group (timestamp = frame index). Coverage computed over the
fully fused set saturates -- almost every pixel is backed by *something* seen
from *somewhere* -- so it cannot tell a view the nearby frames observed from a
view they never looked at. Restricting to the w frames nearest the query
(by the same nearest-position rule that picks the query time) gives a
coverage that falls both when the robot leaves the path and when it turns
away from where those frames looked.

Two knobs, both in CalibratedBackendConfig:
  render_window   > 0: rasterize RGB / depth / labels from the window only
                       (the "static-21" middle ground between static and dynamic)
  coverage_window > 0: compute the coverage statistic from the window only,
                       one extra opacity-only rasterization per step; the
                       image itself may still come from the full fusion

Requires the `source_frame` tag on constant Gaussian sets (NeoVerse
rasterization.py, _create_constant_gaussians). Dynamic per-frame groups are
selected by their own timestamp and need no tag.
"""
from __future__ import annotations

import copy
from typing import List

_warned = set()

# HELD-OUT RANGE (2026-11-25, Jing's held-out test, scripts/heldout_sweep.py): when set to (lo, hi), the window
# keeps source frames lo..hi (inclusive) instead of the frames around t_idx, so a render at a recorded camera can
# leave that frame out and be compared with the real frame. None = the normal centred window; training and
# evaluation never set it.
RANGE = None


def window_gaussians(batch_gaussians: list, t_idx: int, window: int) -> list:
    """Restrict a scene's Gaussian groups (one batch: a list of Gaussians
    objects) to source frames within `window` frames of `t_idx`.

    Constant sets (timestamp -1) are cut by their per-Gaussian `source_frame`
    tag; per-frame groups are kept whole when their timestamp is inside the
    window. Returns a new list; the input objects are not modified.
    """
    import torch
    if window is None or window <= 0:
        return list(batch_gaussians)
    half = max(int(window) // 2, 0)
    lo, hi = (t_idx - half, t_idx + half) if RANGE is None else (int(RANGE[0]), int(RANGE[1]))
    out: List = []
    for g in batch_gaussians:
        ts = int(getattr(g, "timestamp", -1))
        if ts == -1:
            sf = getattr(g, "source_frame", None)
            if sf is None:
                key = id(g)
                if key not in _warned:
                    _warned.add(key)
                    print("[window] constant Gaussian set carries no source_frame tag; "
                          "the window cannot be applied to it (old cache or unpatched "
                          "NeoVerse). Using the full set.", flush=True)
                out.append(g)
                continue
            keep = torch.nonzero((sf >= lo) & (sf <= hi), as_tuple=False).squeeze(-1)
            if keep.numel() == 0:
                continue
            gw = copy.copy(g)             # shallow: keep_indices re-assigns tensors
            gw.keep_indices(keep)
            out.append(gw)
        elif lo <= ts <= hi:
            out.append(g)
    return out


def windowed_alpha(rasterizer, batch_gaussians: list, w2c, K, ts, t_idx: int,
                   window: int, width: int, height: int):
    """Opacity map [H, W] of the windowed Gaussians at one view."""
    sel = window_gaussians(batch_gaussians, int(t_idx), window)
    if not sel:
        return None
    _, _, alpha = rasterizer.forward(
        [sel], render_viewmats=[w2c[None]], render_Ks=[K[None]],
        render_timestamps=[ts[None]], sh_degree=0, width=width, height=height)
    return alpha[0, 0, ..., 0] if alpha.ndim == 5 else alpha[0, 0]


def windowed_coverage(rasterizer, batch_gaussians: list, w2c, K, ts, t_idx: int,
                      window: int, width: int, height: int) -> float:
    """Mean opacity of the windowed render: the coverage statistic the gate reads."""
    a = windowed_alpha(rasterizer, batch_gaussians, w2c, K, ts, t_idx, window, width, height)
    if a is None:
        return 0.0
    return float(a.detach().float().mean().item())
