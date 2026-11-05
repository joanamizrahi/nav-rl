#!/usr/bin/env bash
# Eval episodes drawn on the label map (CPU, login node). 2026-09-27. Output: outputs/overlays_paper/OVERLAY_<evaldir>.png
set -u
cd /scratch/m000204-pm07b/joana/nav-rl
O=/scratch/m000204-pm07b/joana/outputs
python scripts/scene_corner_scan.py quad2_00 --plot outputs/overlays_paper --overlay \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp3000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_120000_steps_goal48_b40p16_v35b_p6_ppl5_e3_quad2_00_live \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp3000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_120000_steps_goal55_b40p16_v35b_p6_ppl5_e3_quad2_00_live \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp1000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_280000_steps_goal48_b40p16_v35b_p6_ppl5_e3_quad2_00_live \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp1000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_280000_steps_goal55_b40p16_v35b_p6_ppl5_e3_quad2_00_live 2>&1 | grep -v WARNING | grep 'wrote\|SKIP\|Error\|Traceback'
python scripts/scene_corner_scan.py sequoia1_21 --plot outputs/overlays_paper --overlay \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp3000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_120000_steps_goal45_b40p16_v35b_p6_ppl5_e3_sequoia1_21_live \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp1000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_280000_steps_goal45_b40p16_v35b_p6_ppl5_e3_sequoia1_21_live 2>&1 | grep -v WARNING | grep 'wrote\|SKIP\|Error\|Traceback'
python scripts/scene_corner_scan.py quad2_04 --plot outputs/overlays_paper --overlay \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp3000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_120000_steps_goal66_b40p16_v35b_p6_ppl5_e3_quad2_04_live \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp1000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_280000_steps_goal66_b40p16_v35b_p6_ppl5_e3_quad2_04_live 2>&1 | grep -v WARNING | grep 'wrote\|SKIP\|Error\|Traceback'
python scripts/scene_corner_scan.py gnd_AU_180 --clips /scratch/m000204-pm07b/joana/data/gnd_clips --plot outputs/overlays_paper --overlay \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp3000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_120000_steps_goal32_b40p16_v35b_p6_ppl5_e3_gnd_AU_180_live \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp1000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_280000_steps_goal32_b40p16_v35b_p6_ppl5_e3_gnd_AU_180_live 2>&1 | grep -v WARNING | grep 'wrote\|SKIP\|Error\|Traceback'
python scripts/scene_corner_scan.py gnd_AUw360 --clips /scratch/m000204-pm07b/joana/data/gnd_clips --plot outputs/overlays_paper --overlay \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp3000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_120000_steps_goal48_b40p16_v35b_p6_ppl5_e3_gnd_AUw360_live \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp1000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_280000_steps_goal48_b40p16_v35b_p6_ppl5_e3_gnd_AUw360_live 2>&1 | grep -v WARNING | grep 'wrote\|SKIP\|Error\|Traceback'
python scripts/scene_corner_scan.py gnd_AUw390 --clips /scratch/m000204-pm07b/joana/data/gnd_survey --plot outputs/overlays_paper --overlay \
    $O/eval_x4_rw21_cw21_g4_ms13_prox5_warmcheckpoints_rw5_cp1000_ms90_gsup0.6_bmamean_bm5_static_shw_sf_mir0.5_r1.0_la_nh_ppo_280000_steps_goal48_b40p16_v35b_p6_ppl5_e3_gnd_AUw390_live 2>&1 | grep -v WARNING | grep 'wrote\|SKIP\|Error\|Traceback'
