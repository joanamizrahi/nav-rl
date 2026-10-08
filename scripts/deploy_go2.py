"""Deploy a trained policy on the Go2W: camera + odometry in, velocity out.

Publishes geometry_msgs/Twist on /cmd_vel — the input of soar-go2's
motion_control.py (go2w_sdk), which owns the Unitree SDK and enforces a
deadman timeout + /estop. Run motion_control first; this node replaces the
teleop source. (/joystick_cmd_vel is only joy_bridge's RECORDING stream —
nothing drives the robot from it.) No diffusion runs on the robot — only the
small PPO network; the measured per-inference milliseconds are logged.

The policy decides at --rate Hz, but the last command is REPUBLISHED at 20 Hz
so motion_control's deadman never trips between decisions.

Loop at --rate Hz:
  1. latest camera frame  -> center-crop 5:3 -> resize 560x336  (same
     preprocessing as prepare_rosbag_clips.py, so the policy sees the
     training distribution)
  2. goal vector: the goal is fixed in the odom frame at startup
     (--goal_dx/--goal_dy meters, robot frame at t=0); each tick it is
     re-expressed in the CURRENT robot frame from /Odometry -> (dx, dy, bearing)
  3. action = policy(rgb, goal); v = a0 * 0.25 * rate, w = a1 * 0.3 * rate
     (the sim step is 0.25 m / 0.3 rad; scaling by rate converts steps/s to
     m/s and rad/s), clamped to --max_v / --max_w
  4. publish Twist on /cmd_vel (kept alive at 20 Hz for the deadman)

Safety: --dry_run computes but never publishes. /estop (std_msgs/Bool) is
motion_control's kill switch — keep a terminal ready with:
  ros2 topic pub /estop std_msgs/msg/Bool "data: true" --once
Zero command is sent when the goal is within 0.75 m and on Ctrl-C.

Usage on the Jetson:
    python3 scripts/deploy_go2.py \
        --checkpoint ppo_800000_steps.zip --goal_dx 5.0 --goal_dy 0.0 \
        [--rate 2.0] [--dry_run]
"""

from __future__ import annotations
import os as _os0; _os0.environ.setdefault("XFORMERS_DISABLED", "1")  # Thor 2026-10-08: xFormers has no kernel for this GPU; DINOv2 falls back to plain attention

import argparse
import json
import os
import time

import numpy as np


def quat_to_yaw(x, y, z, w) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# The DINO policies pickle src.policy.encoders.FrozenBackboneExtractor, so `src`
# must be importable: from the repo (scripts/../src) or from a copy of src/ placed
# next to this file on Thor (~/nav_policy/src). 2026-10-07.
import os as _os, sys as _sys
for _p in (_os.path.dirname(_os.path.abspath(__file__)),
           _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

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

    _sys.meta_path.insert(0, _NumpyCoreAlias())

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

def preprocess(bgr: np.ndarray, W: int = 560, H: int = 336) -> np.ndarray:
    import cv2
    h, w = bgr.shape[:2]
    ar = W / H
    ch = int(w / ar)
    if ch <= h:
        y0 = (h - ch) // 2
        crop = bgr[y0:y0 + ch]
    else:
        cw = int(h * ar)
        x0 = (w - cw) // 2
        crop = bgr[:, x0:x0 + cw]
    rgb = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    return np.ascontiguousarray(rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="policy .zip (required unless --baseline)")
    ap.add_argument("--baseline", action="store_true",
                    help="IGNORE the camera: pure goal-bearing servo "
                         "(v ~ distance, w ~ bearing). The control lap for "
                         "the outdoor comparison — any difference between this "
                         "trajectory and the policy's is what world-model "
                         "training contributed.")
    ap.add_argument("--goal_dx", type=float, required=True,
                    help="goal x (m) in the robot frame at startup (forward)")
    ap.add_argument("--goal_dy", type=float, required=True,
                    help="goal y (m) in the robot frame at startup (left)")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--max_v", type=float, default=0.6)
    ap.add_argument("--max_w", type=float, default=0.8)
    ap.add_argument("--goal_radius", type=float, default=None,
                    help="stop within this distance; defaults to the run's "
                         "goal_radius from env_config.json, else 0.75 m")
    ap.add_argument("--image_topic", default="/odin1/image/undistorted",
                    help="camera topic; CompressedImage if it ends in /compressed, "
                         "else raw Image (2026-09-09: Odin camera on Thor)")
    ap.add_argument("--odom_topic", default="/Odometry")
    ap.add_argument("--cmd_topic", default="/cmd_vel")
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="command blending 0..1: published = smooth*new + "
                         "(1-smooth)*previous. 1.0 = raw policy output; lower "
                         "= gentler transitions (soar-go2's RL rate-limits "
                         "velocity changes the same way in training)")
    ap.add_argument("--fixed_goal", default=None,
                    help='DIAGNOSTIC, dry-run only: "dx,dy" held CONSTANT in the '
                         "robot frame regardless of odometry, so any variation in "
                         "the action comes from the image alone")
    ap.add_argument("--swap_rb", action="store_true",
                    help="flip red and blue: use when check_camera_topic.py shows "
                         "an orange sky (stream is RGB where BGR was assumed)")
    ap.add_argument("--timeout_s", type=float, default=60.0,
                    help="abort and stop the robot after this long (0 disables)")
    ap.add_argument("--no_progress_s", type=float, default=15.0,
                    help="abort if the goal distance has not improved by 0.25 m "
                         "within this many seconds -- catches circling (0 disables)")
    ap.add_argument("--log", default=None,
                    help="write a CSV of every decision for later analysis")
    ap.add_argument("--step_m", type=float, default=0.25,
                    help="metres per decision at |a0|=1; overridden by the run's env_config.json")
    ap.add_argument("--yaw_rad", type=float, default=0.30,
                    help="radians per decision at |a1|=1; overridden by the run's env_config.json")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.fixed_goal is not None and not args.dry_run:
        ap.error("--fixed_goal is a diagnostic and requires --dry_run: the goal "
                 "would not correspond to anywhere real.")
    args.goal_radius_set = args.goal_radius is not None
    if args.goal_radius is None:
        args.goal_radius = 0.75

    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from stable_baselines3 import PPO

    # The observation size and the action scaling MUST match training. Both are
    # recorded by the run itself, so read them rather than hardcoding: the
    # render size (560x336) and the observation size (336x224) are different
    # numbers and confusing them silently feeds the policy the wrong picture.
    OBS_W, OBS_H = 560, 336
    step_m, yaw_rad = args.step_m, args.yaw_rad
    if args.baseline:
        model = None
        print("[deploy] BASELINE mode: goal-bearing servo, camera ignored")
    else:
        assert args.checkpoint, "--checkpoint required unless --baseline"
        model = PPO.load(args.checkpoint, device="cuda")
        sp = model.observation_space
        assert "rgb" in sp.spaces and "goal" in sp.spaces, f"unexpected obs space {sp}"
        # SB3 stores the space channels-FIRST when training went through
        # VecTransposeImage (ours does): (3, H, W). Older runs may be (H, W, 3).
        # predict() accepts either layout at call time, but the SIZE must be
        # read from the right axes or the camera frame gets resized to 3 px.
        shp = tuple(int(v) for v in sp["rgb"].shape)
        assert len(shp) == 3, f"unexpected rgb shape {shp}"
        OBS_H, OBS_W = (shp[1], shp[2]) if shp[0] in (1, 3) else (shp[0], shp[1])
        assert tuple(sp["goal"].shape) == (3,), f"goal space is {sp['goal'].shape}, expected (3,)"
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(args.checkpoint))), "env_config.json")
        src = "CLI default"
        fwd_only = False
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                envc = json.load(fh)
            if "step_size_m" in envc:  step_m = float(envc["step_size_m"]); src = cfg_path
            if "yaw_step_rad" in envc: yaw_rad = float(envc["yaw_step_rad"])
            # Match the radius the policy was trained to stop inside; a tighter
            # one asks for precision it never learned.
            if "goal_radius" in envc and not args.goal_radius_set:
                args.goal_radius = float(envc["goal_radius"])
            # 2026-09-28: every policy since 09-07 trained forward_only (a negative throttle
            # sample meant "stand still", never "reverse"). Mirror that here, or a negative
            # mean action would drive the robot backwards into something it never saw.
            fwd_only = bool(envc.get("forward_only", False))
        print(f"[deploy] loaded {args.checkpoint}")
        print(f"[deploy] observation from checkpoint: {OBS_W}x{OBS_H} (WxH)")
        print(f"[deploy] step {step_m} m, yaw {yaw_rad} rad per decision  [{src}]")
        print(f"[deploy] at {args.rate} Hz -> max v {step_m * args.rate:.2f} m/s, "
              f"max w {yaw_rad * args.rate:.2f} rad/s (clipped to {args.max_v}/{args.max_w})")
        print(f"[deploy] stop radius {args.goal_radius:.2f} m")
        if args.max_v < step_m * args.rate - 1e-6:
            print(f"[deploy] WARNING: max_v {args.max_v} clips the policy's full-forward "
                  f"command ({step_m * args.rate:.2f} m/s). It will move less than it "
                  f"expects each decision. Prefer lowering --rate to go slower.")

    class PolicyNode(Node):
        def __init__(self):
            super().__init__("nav_policy")
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            # Accept either transport: a raw Image topic with a CompressedImage
            # subscription never fires a callback and never errors, so the node
            # would just wait forever. Rule matches check_camera_topic.py.
            self._compressed = args.image_topic.endswith("/compressed")
            img_type = CompressedImage if self._compressed else Image
            print(f"[deploy] image topic {args.image_topic} as {img_type.__name__}")
            self.create_subscription(img_type, args.image_topic, self.on_img, qos)
            self.create_subscription(Odometry, args.odom_topic, self.on_odom, 10)
            self.pub = self.create_publisher(Twist, args.cmd_topic, 10)
            self.img = None
            self.pose = None            # (x, y, yaw) in odom frame
            self.goal_odom = None       # set on first odom using startup frame
            self.lat = []
            self.cmd = (0.0, 0.0)       # latest decided (v, w)
            self.t_start = time.time()
            self.best_dist = float("inf")
            self.t_best = time.time()
            self.rows = []              # per-decision log for post-hoc analysis
            self.done = False
            self.create_timer(1.0 / args.rate, self.tick)
            # deadman keep-alive: motion_control stops if /cmd_vel goes quiet,
            # so re-send the current command at 20 Hz between policy decisions
            self.create_timer(0.05, self.keepalive)

        def on_img(self, msg):
            # Always store BGR, because preprocess() flips BGR -> RGB.
            if self._compressed:
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, np.uint8).reshape(
                    msg.height, msg.width, -1)[:, :, :3]
                if str(msg.encoding).lower().startswith("rgb"):
                    img = img[:, :, ::-1]
            self.img = img[:, :, ::-1] if args.swap_rb else img

        def on_odom(self, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            self.pose = (p.x, p.y, yaw)
            if self.goal_odom is None:
                c, s = np.cos(yaw), np.sin(yaw)
                self.goal_odom = (p.x + c * args.goal_dx - s * args.goal_dy,
                                  p.y + s * args.goal_dx + c * args.goal_dy)
                self.get_logger().info(f"goal fixed at odom ({self.goal_odom[0]:.2f}, "
                                       f"{self.goal_odom[1]:.2f})")

        def abort(self, why):
            self.done = True
            self.publish(0.0, 0.0)
            self.get_logger().error(f"ABORT: {why} -- robot stopped")

        def tick(self):
            if self.done:
                self.publish(0.0, 0.0)
                return
            if self.img is None or self.pose is None or self.goal_odom is None:
                self.get_logger().info("waiting for camera/odom ...", throttle_duration_sec=2.0)
                return
            x, y, yaw = self.pose
            dxw, dyw = self.goal_odom[0] - x, self.goal_odom[1] - y
            dist = float(np.hypot(dxw, dyw))
            c, s = np.cos(-yaw), np.sin(-yaw)
            dx, dy = c * dxw - s * dyw, s * dxw + c * dyw   # goal in robot frame
            if args.fixed_goal is not None:
                # Hold the goal input constant so only the image varies.
                dx, dy = (float(q) for q in args.fixed_goal.split(","))
                dist = float(np.hypot(dx, dy))
            bearing = float(np.arctan2(dy, dx))
            if dist < args.goal_radius:
                self.done = True
                self.publish(0.0, 0.0)
                self.get_logger().info(f"GOAL REACHED ({dist:.2f} m) — holding")
                return
            # Circling and stalling guards. The corner evals in simulation showed
            # loops and overshoots, so do not rely on watching it by eye.
            now = time.time()
            if dist < self.best_dist - 0.25:
                self.best_dist, self.t_best = dist, now
            if args.timeout_s > 0 and now - self.t_start > args.timeout_s:
                return self.abort(f"timeout after {args.timeout_s:.0f} s")
            if args.no_progress_s > 0 and now - self.t_best > args.no_progress_s:
                return self.abort(f"no progress for {args.no_progress_s:.0f} s "
                                  f"(best {self.best_dist:.2f} m, now {dist:.2f} m)")
            t0 = time.perf_counter()
            if args.baseline:
                v = float(np.clip(0.5 * dist, 0.0, args.max_v))
                w = float(np.clip(1.5 * bearing, -args.max_w, args.max_w))
            else:
                obs = {"rgb": preprocess(self.img, OBS_W, OBS_H),
                       "goal": np.array([dx, dy, bearing], dtype=np.float32)}
                action, _ = model.predict(obs, deterministic=True)
                v = float(np.clip(action[0] * step_m * args.rate, -args.max_v, args.max_v))
                if fwd_only:
                    v = max(0.0, v)
                w = float(np.clip(action[1] * yaw_rad * args.rate, -args.max_w, args.max_w))
            ms = (time.perf_counter() - t0) * 1e3
            self.lat.append(ms)
            a = float(np.clip(args.smooth, 0.0, 1.0))
            v = a * v + (1.0 - a) * self.cmd[0]
            w = a * w + (1.0 - a) * self.cmd[1]
            self.publish(v, w)
            self.rows.append((now - self.t_start, x, y, yaw, dist, bearing, v, w, ms))
            self.get_logger().info(
                f"dist {dist:4.1f} m bearing {np.degrees(bearing):+5.0f} deg | "
                f"v {v:+.2f} w {w:+.2f} | policy {ms:.1f} ms "
                f"(median {np.median(self.lat):.1f})")

        def publish(self, v, w):
            self.cmd = (v, w)
            self.keepalive()

        def keepalive(self):
            if args.dry_run:
                return
            msg = Twist()
            msg.linear.x = self.cmd[0]
            msg.angular.z = self.cmd[1]
            self.pub.publish(msg)

    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
        except Exception:
            pass    # Ctrl-C can tear down the ROS context before this runs
        if node.lat:
            print(f"[deploy] policy latency: median {np.median(node.lat):.1f} ms "
                  f"over {len(node.lat)} inferences")
        if args.log and node.rows:
            import csv
            with open(args.log, "w", newline="") as fh:
                wtr = csv.writer(fh)
                wtr.writerow(["t", "x", "y", "yaw", "dist", "bearing", "v", "w", "ms"])
                wtr.writerows(node.rows)
            print(f"[deploy] wrote {len(node.rows)} decisions to {args.log}")
        try:
            rclpy.shutdown()          # Ctrl-C already shut the context down
        except Exception:
            pass


if __name__ == "__main__":
    main()
