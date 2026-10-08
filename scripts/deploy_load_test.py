"""Load every policy under ~/nav_policy (or the dirs given) exactly as deploy_go2.py will,
run one forward pass on a random frame, and report. Run on the robot computer BEFORE the
field day; a DINO policy that cannot import src/ or download its backbone fails here, not
on the lawn.   python3 deploy_load_test.py [run_dir ...]
"""
import glob, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np
from stable_baselines3 import PPO
runs = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join(HERE, "*")) if os.path.isdir(os.path.join(d, "checkpoints")))
ok = 0
for r in runs:
    zs = sorted(glob.glob(os.path.join(r, "checkpoints", "ppo_*_steps.zip")))
    if not zs: print(f"{os.path.basename(r)}: no checkpoint"); continue
    try:
        t = time.time(); m = PPO.load(zs[-1], device="cuda"); shp = tuple(int(v) for v in m.observation_space["rgb"].shape)
        H, W = (shp[1], shp[2]) if shp[0] in (1, 3) else (shp[0], shp[1])
        obs = {"rgb": np.random.randint(0, 255, (H, W, 3), np.uint8), "goal": np.array([3.0, 0.0, 0.0], np.float32)}
        a, _ = m.predict(obs, deterministic=True); t1 = time.time()
        for _ in range(20): m.predict(obs, deterministic=True)
        print(f"{os.path.basename(r)}: OK  obs {W}x{H}  action {np.round(a, 3)}  load {t1-t:.1f}s  {1000*(time.time()-t1)/20:.1f} ms/step  env_config {'yes' if os.path.exists(os.path.join(r, 'env_config.json')) else 'MISSING'}")
        ok += 1
    except Exception as e:
        print(f"{os.path.basename(r)}: FAILED  {type(e).__name__}: {str(e)[:200]}")
print(f"{ok}/{len(runs)} policies load")
