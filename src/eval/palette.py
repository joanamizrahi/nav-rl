"""Canonical color palette for the 30-class Go2W outdoor taxonomy.

Mirrors NeoVerse's `diffsynth/utils/semantics.CLASS_COLORS` so we don't need
torch on Mac to render label overlays. Values are (R, G, B) uint8 in [0, 255].

LOCKSTEP: if NeoVerse's CLASS_COLORS changes, update this table too.
"""
from __future__ import annotations
import os as _os
_S = _os.environ.get("SCRATCH_ROOT", "/scratch/m000204-pm07b/joana")  # 2026-11-05: Cycle 7 scratch; override with SCRATCH_ROOT

import numpy as np


CLASS_COLORS_255 = np.array([
    (  0,   0,   0),   # 0  void
    (200, 225, 245),   # 1  sky
    (139,  90,  43),   # 2  dirt
    (230, 200, 155),   # 3  sand
    ( 75, 190,  80),   # 4  grass
    (180, 155, 100),   # 5  gravel
    (110,  55,  25),   # 6  mulch
    ( 55,  55,  30),   # 7  mud
    ( 50, 120, 200),   # 8  water
    (135, 145, 155),   # 9  rock
    ( 55,  55,  65),   # 10 asphalt
    (225, 220, 190),   # 11 concrete
    (110, 110, 115),   # 12 road
    (180, 180, 180),   # 13 sidewalk
    (255, 250, 235),   # 14 crosswalk
    (170,  75,  60),   # 15 building
    (175, 145, 175),   # 16 wall
    ( 90,  60, 130),   # 17 fence
    ( 75, 155, 175),   # 18 bridge
    ( 40, 105,  55),   # 19 tree
    (170, 200,  55),   # 20 vegetation
    (135, 115,  90),   # 21 log
    (220, 140,  80),   # 22 stairs
    ( 25,  65, 130),   # 23 pole
    (230, 195,  60),   # 24 traffic_sign
    (235,  85,  75),   # 25 traffic_light
    (110, 130, 220),   # 26 vehicle
    (155,  60, 200),   # 27 motorcycle
    (100, 230, 200),   # 28 bicycle
    (205,  70, 145),   # 29 person
], dtype=np.uint8)

# v14 navigation taxonomy — LOCKSTEP mirror of NeoVerse
# diffsynth/utils/class_taxonomy.V14 (colors only; scores live in
# config/traversability_v14.yaml). Cache labels and SAM3 v14 npz use THESE ids.
CLASS_COLORS_V14_255 = np.array([
    (  0,   0,   0),   # 0  void
    (200, 225, 245),   # 1  sky
    (150, 100,  55),   # 2  trail (dirt/gravel/mulch)
    ( 75, 190,  80),   # 3  grass
    ( 95,  65,  35),   # 4  rough (sand/mud/rock/log)
    ( 50, 120, 200),   # 5  water
    (210, 210, 210),   # 6  sidewalk
    ( 70,  70,  85),   # 7  road
    (235, 205, 150),   # 8  pavement-unknown
    (220, 140,  80),   # 9  stairs
    (185,  55,  50),   # 10 obstacle
    (170, 200,  55),   # 11 vegetation
    (205,  70, 145),   # 12 person
    (110, 130, 220),   # 13 vehicle
], dtype=np.uint8)


def display_palette(sem_palette: int = 4, strict: bool = False) -> np.ndarray:
    """[14,3] uint8 colors for label ids, the SAME rule save_rollout_video uses
    for the eval videos (2026-09-06, Joana: the side panels must match the
    sighted/blind videos): diffsynth's v14_palette(sem_palette) when the
    NeoVerse tree is importable, else the static CLASS_COLORS_V14_255.
    strict=True raises instead of falling back, and the module-level
    DISPLAY_PALETTE_SOURCE says which one was used."""
    global DISPLAY_PALETTE_SOURCE
    err = None
    try:
        import sys
        from pathlib import Path
        for _nv in (Path(__file__).resolve().parents[2].parent / "NeoVerse",
                    Path(_S + "/NeoVerse")):
            if _nv.exists() and str(_nv) not in sys.path:
                sys.path.insert(0, str(_nv))
        from diffsynth.utils.class_taxonomy import v14_palette
        pal = v14_palette(int(sem_palette))
        pal = pal.numpy() if hasattr(pal, "numpy") else np.asarray(pal)
        pal = (pal * 255).astype(np.uint8) if pal.max() <= 1.0 else pal.astype(np.uint8)
        if pal.ndim == 2 and pal.shape[1] == 3 and len(pal) >= 14:
            DISPLAY_PALETTE_SOURCE = f"diffsynth v14_palette({int(sem_palette)})"
            return pal[:14]
        err = f"unexpected palette shape {pal.shape}"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    DISPLAY_PALETTE_SOURCE = f"STATIC CLASS_COLORS_V14_255 (fallback: {err})"
    if strict:
        raise RuntimeError(f"display_palette({sem_palette}) could not load the diffsynth palette: {err}")
    return CLASS_COLORS_V14_255


DISPLAY_PALETTE_SOURCE = "unset"


CLASS_NAMES_V14 = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk", "road",
                   "pavement?", "stairs", "obstacle", "vegetation", "person", "vehicle"]
