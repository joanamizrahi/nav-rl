"""Load every policy under ~/nav_policy (or the dirs given) exactly as deploy_go2.py will,
run one forward pass on a random frame, and report. Run on the robot computer BEFORE the
field day; a DINO policy that cannot import src/ or download its backbone fails here, not
on the lawn.   python3 deploy_load_test.py [run_dir ...]
"""
import os as _os0; _os0.environ.setdefault("XFORMERS_DISABLED", "1")  # Thor 2026-10-08: xFormers has no kernel for this GPU; DINOv2 falls back to plain attention

import glob, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np
# Checkpoints trained under numpy 2.x pickle references to `numpy._core.*`;
# Thor runs numpy 1.26 (ROS/cv2 pin), where that package is `numpy.core`.
# Alias it so cloudpickle can resolve them. 2026-10-08.
if not np.__version__.startswith("2"):
    import importlib as _il, importlib.abc as _ilabc, importlib.util as _ilutil

    class _NumpyCoreAlias(_ilabc.MetaPathFinder, _ilabc.Loader):
        def find_spec(self, name, path=None, target=None):
            if name == "numpy._core" or name.startswith("numpy._core."):
                return _ilutil.spec_from_loader(name, self)
            return None

        def create_module(self, spec):
            return _il.import_module("numpy.core" + spec.name[len("numpy._core"):])

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _NumpyCoreAlias())

    # numpy 2.x also pickles a space's RNG differently: the BitGenerator is
    # passed as a class (1.26 wants its name), its state as (state, seed_seq)
    # (1.26 wants the dict), and the Generator as a BitGenerator instance.
    import numpy.random._pickle as _nrp
    _orig_bg_ctor = _nrp.__bit_generator_ctor
    _bg_compat = {}

    def _bg_ctor(bit_generator_name="MT19937"):
        if not isinstance(bit_generator_name, type):
            return _orig_bg_ctor(bit_generator_name)
        cls = bit_generator_name
        if cls not in _bg_compat:
            def __setstate__(self, state):
                self.state = state[0] if isinstance(state, tuple) else state
            _bg_compat[cls] = type(cls.__name__, (cls,), {"__setstate__": __setstate__})
        return _bg_compat[cls]()

    def _gen_ctor(bit_generator_name="MT19937", bit_generator_ctor=_bg_ctor):
        if isinstance(bit_generator_name, np.random.BitGenerator):
            return np.random.Generator(bit_generator_name)
        return np.random.Generator(bit_generator_ctor(bit_generator_name))

    _nrp.__bit_generator_ctor = _bg_ctor
    _nrp.__generator_ctor = _gen_ctor
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
        ms = 1000 * (time.time() - t1) / 20
        # the same image check the cluster evals print: action on a black frame vs on noise.
        # ~0.35-0.45 for the policies we deploy; below 0.05 means the policy ignores the camera.
        black = {"rgb": np.zeros((H, W, 3), np.uint8), "goal": obs["goal"]}
        d = float(np.abs(m.predict(black, deterministic=True)[0] - m.predict(obs, deterministic=True)[0]).sum())
        print(f"{os.path.basename(r)}: OK  obs {W}x{H}  {ms:.1f} ms/step  image check |a(noise)-a(black)| = {d:.3f} "
              f"({'uses the image' if d > 0.05 else 'IGNORES THE IMAGE'})  env_config {'yes' if os.path.exists(os.path.join(r, 'env_config.json')) else 'MISSING'}")
        ok += 1
    except Exception as e:
        print(f"{os.path.basename(r)}: FAILED  {type(e).__name__}: {str(e)[:200]}")
print(f"{ok}/{len(runs)} policies load")
