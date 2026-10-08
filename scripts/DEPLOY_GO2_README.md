# Deploying the nav policy on the Go2W — runbook

The policy is a small network trained inside the diffusion world model. On the
robot it is only that network (~8 ms per inference on Thor); **no diffusion runs
at deploy time**. It takes a camera frame and a relative goal, and outputs a
forward and a yaw command on `/cmd_vel`.

Follow the sections in order. Do not skip section 3.

Files on Thor live in `~/nav_policy/`:

```
~/nav_policy/
  deploy_go2.py                        the policy node
  <run_name>/
    checkpoints/ppo_XXXXXX_steps.zip   the policy
    env_config.json                    how it was trained (read automatically)
```

`env_config.json` must sit one directory above `checkpoints/`, because that is
where the node looks for it to recover the step size and yaw step.

---

> **Topic names on the Go2W as recorded 2026-09-14:** camera `/odin1/image/undistorted` (raw Image, ~13 Hz) or `/camera/camera/color/image_raw/compressed` (RealSense, ~30 Hz); odometry `/odin1/odometry` (the only nav_msgs/Odometry topic; pass `--odom_topic /odin1/odometry`, the script default is `/Odometry`). Camera heights: Odin 0.69 m, RealSense 0.50 m.

## 1. Before you leave the lab

**Copy the current code.** `deploy_go2.py` changes; an old copy on Thor will
silently use the wrong image size.

```bash
scp scripts/deploy_go2.py soar@<thor>:~/nav_policy/
```

**Copy the policy and its config together**, from wherever the run lives:

```bash
scp "marlowe:/scratch/m000204-pm06b/joana/outputs/<run>/checkpoints/ppo_XXXXXX_steps.zip" .
scp "marlowe:/scratch/m000204-pm06b/joana/outputs/<run>/env_config.json" .
scp -r <run> soar@<thor>:~/nav_policy/
```

**Write down what you are testing and what would count as success** before the
robot is switched on: the goal distance, the route you expect, and the failure
you are watching for. A run nobody predicted is hard to learn from.

---

## 2. Bringup

1. Robot on, **terrain mode** on the controller.
2. **Orin** (`ssh -J soar@<thor> unitree@192.168.123.18`, pw `123`), three terminals
   (2026-10-08: this is the lab's current sequence, from
   `local_map_baselines/docs/outdoor_test_commands.md`; the old
   `python3 go2w_sdk/motion_control.py` path no longer exists):
   ```bash
   cd ~ && ros2 run domain_bridge domain_bridge bridge_config.yaml
   ros2 launch livox_ros_driver2 msg_MID360_launch.py
   ros2 launch go2w_sdk motion_control.launch.py config_file:=$HOME/navigation_ws/src/soar-go2/go2w_sdk/configs/sdk.yaml
   ```
3. **Thor** — camera. The Odin runs in a Docker container; after every boot
   first `echo 100 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb`, then:
   ```bash
   cd ~/Documents/navigation_ws/src/navigation_pipeline/ && docker run -it --rm --privileged --net=host \
       -e CYCLONEDDS_URI='<CycloneDDS><Domain><General><AllowMulticast>spdp</AllowMulticast></General></Domain></CycloneDDS>' \
       -v /dev/bus/usb:/dev/bus/usb -v $(pwd)/data/odin1_interface:/root/odin1_interface \
       -v $(pwd)/data/odin1_config:/root/ros2_ws/src/odin_ros_driver/config odin1:v1
   # inside the container:
   ros2 launch odin_ros_driver odin1_ros2.launch.py
   # second Thor terminal:
   ros2 launch local_map_baselines frame_transformation_odin.launch.py
   ```
   Do NOT start the lab's motion planner (`motion_planner_runner.py`) or the
   perception stack for a policy test: this node publishes `/cmd_vel` itself.
4. **Thor** — an estop terminal, command pre-typed and not yet run:
   ```bash
   ros2 topic pub /estop std_msgs/msg/Bool "data: true" --once
   ```

**Every terminal that runs the policy** needs the conda path stripped, or
conda's numpy 1 shadows the venv's numpy 2 and the checkpoint will not load:

```bash
export PYTHONPATH=$(echo "$PYTHONPATH" | sed 's#:/home/soar/miniconda3/lib/python3.12/site-packages##')
```

The `~/nav_env` venv exists for that reason (created 2026-08-23): system
site-packages plus numpy 2, `stable-baselines3` installed with `--no-deps`,
plus gymnasium, cloudpickle and pandas. Run the policy with
`~/nav_env/bin/python3`, never the system python. To rebuild it:

```bash
python3 -m venv --system-site-packages ~/nav_env
~/nav_env/bin/pip install "numpy>=2"
~/nav_env/bin/pip install --no-deps stable-baselines3
~/nav_env/bin/pip install gymnasium cloudpickle pandas
```

---

## 3. Pre-flight, every session

**Both topics must tick.** No output means no data, and the node waits forever:

```bash
ros2 topic hz /camera/camera/color/image_raw/compressed
ros2 topic hz /Odometry
```

**Check the camera topic**, always after a camera change:

```bash
ros2 topic list | grep -i -E "image|color|odin"
ros2 topic info /odin1/image/undistorted

~/nav_env/bin/python3 check_camera_topic.py \
    --topic /odin1/image/undistorted \
    --out ~/nav_policy/frame_check.png \
    --policy_out ~/nav_policy/frame_policy_view.png
```

(2026-09-09: the camera on Thor is `/odin1/image/undistorted`, a raw
`sensor_msgs/msg/Image`, `bgr8`, 1600x1296. Aspect 1.23 against the policy's
1.67, so the crop keeps the FULL WIDTH and trims top and bottom to about 74%
of the vertical field of view. Pass the same `--image_topic` to
`deploy_go2.py`. Since 2026-09-09 it is also the DEFAULT in both
scripts, so `--image_topic` only needs passing if you switch back to the
RealSense.)

It prints the encoding, the resolution and aspect, the per-channel means, and
writes three images: the full frame, `frame_policy_view.png` which is the exact
336x224 the network receives, and `frame_policy_view_2x.png` which is the same
doubled so it is viewable. **Look at the policy view**, not just the full frame:
the horizon should sit near the middle with a good amount of ground in the lower
half, like the training footage. A horizon very high or very low means the
camera is framed differently from what the policy trained on, and that matters
more than any flag. On colour: grass green and sky blue means the decode is
right; an orange sky means the channels are swapped and you should add
`--swap_rb` to the deploy command.

Why this check exists, and what it protects against:

- **Wrong transport.** `Image` is the raw pixel array (~900 kB a frame),
  `CompressedImage` is the JPEG (~50 kB). Both scripts pick the type from the
  topic name: ending in `/compressed` means `CompressedImage`, otherwise
  `Image`. Subscribing with the wrong type fires no callback and raises no
  error, so the node would sit on "waiting for camera/odom" forever.
- **Wrong colour order.** `cv2.imdecode` returns BGR and the node flips it to
  RGB. A raw topic's `encoding` field is honoured automatically; anything else
  else odd is what `--swap_rb` is for. Swapped channels do not crash, they just make
  the policy behave strangely.
- **Field of view.** The policy trained on the field of view of the original
  walk footage. `preprocess` center-crops to 5:3 and resizes, which fixes shape
  but not FOV. A much wider or narrower camera shows the world at a different
  scale than the policy expects, and that is a better explanation for odd
  behaviour than a bad policy.

Topics are flags on the deploy node (`--image_topic`, `--odom_topic`,
`--cmd_topic`), so pass the new one rather than editing code.

**Dry run.** Computes and prints commands, publishes nothing, robot cannot
move:

```bash
cd ~/nav_policy
~/nav_env/bin/python3 deploy_go2.py --checkpoint <run>/checkpoints/ppo_XXXXXX_steps.zip \
    --goal_dx 3.0 --goal_dy 0.0 --dry_run
```

**Read these three lines before continuing:**

```
[deploy] observation from checkpoint: 336x224 (WxH)
[deploy] step 0.25 m, yaw 0.3 rad per decision  [.../env_config.json]
[deploy] at 2.0 Hz -> max v 0.50 m/s, max w 0.60 rad/s
```

- The resolution comes from the checkpoint itself. 560x336 is the *render* size
  and 336x224 is the *observation* size; an older copy of this node hardcoded
  the former, which fed recent policies a differently framed picture than they
  trained on.
- If the second line says `[CLI default]`, `env_config.json` was not found and
  the action scaling is a guess. Go back and copy it.
- The third line is the speed you are about to allow. Decide it is acceptable
  for the space you are standing in.

Also check the goal line the node prints. The goal is (`--goal_dx` forward,
`--goal_dy` left) of the robot's pose **at node start**, then frozen in the odom
frame. If the robot is not pointing where you think, the goal is not where you
think.

---

## 4. Run

**Always run the baseline first.** It ignores the camera and drives to the goal
on odometry alone. It verifies odometry, the goal frame, the deadman keep-alive
and your safe distances, with no network in the loop. It is also the control
your results need: if the baseline reaches the goal too, the task did not
require vision.

```bash
~/nav_env/bin/python3 deploy_go2.py --baseline \
    --goal_dx 4.0 --goal_dy 0.0 --max_v 0.4 --log baseline_1.csv
```

**Then the policy, on the identical goal:**

```bash
~/nav_env/bin/python3 deploy_go2.py --checkpoint <run>/checkpoints/ppo_XXXXXX_steps.zip \
    --goal_dx 4.0 --goal_dy 0.0 \
    --rate 2 --max_v 0.5 --max_w 0.6 --smooth 0.5 \
    --timeout_s 60 --no_progress_s 15 --log policy_1.csv
```

Record video and topics for anything you may want to show or analyse:

```bash
ros2 bag record /camera/camera/color/image_raw/compressed /Odometry /cmd_vel -o nav_policy_test
```

Start easy and add difficulty one step at a time: open pavement straight ahead,
then a goal with grass on one side, then a goal past a bend. Change one thing
per run.

---

## 5. Stopping

- **Ctrl-C** in the policy terminal sends a zero command and exits.
- **Estop** topic in the pre-typed terminal.
- **`--timeout_s`** stops the robot after that many seconds (default 60).
- **`--no_progress_s`** stops it if the goal distance has not improved by 0.25 m
  within that window (default 15). This is the circling guard: in simulation the
  policy looped and overshot on corner goals, and circling is hard to call by
  eye in the first few seconds.
- If the node dies, motion_control's own deadman stops the robot because
  `/cmd_vel` goes quiet.

---

## 6. After each run

Keep the `--log` CSV and the bag, and write one line about what happened while
it is fresh. The CSV has `t, x, y, yaw, dist, bearing, v, w, ms` for every
decision, which is enough to tell circling from oscillation from driving into
something, without relying on memory.

---

## 7. What the policy actually outputs, and how to choose the rate

The action is **unitless**, two numbers in [-1, 1]. The code names them
`(v_forward, omega_yaw)`, which reads like velocity, but the training
environment applies them as a fixed **displacement** with no clock at all
(`_advance_pose` in `src/env/scene_env.py`): turn by `action[1] * yaw_step_rad`,
then move `action[0] * step_size_m`, instantly. One decision is one step, and a
step has no duration.

So "1.0 forward" has no speed until you decide how long a decision lasts, and
that is exactly what `--rate` sets:

    v = action * step_size_m * rate

At 2 Hz a decision lasts 0.5 s, so a 0.25 m step is 0.5 m/s. At 1 Hz the same
action is 0.25 m/s. The policy is indifferent: in its world it ends up 0.25 m
further along either way.

Two consequences:

- **Slow down with `--rate`, not `--max_v`.** A lower rate keeps every step at
  0.25 m and just takes longer. Clipping the velocity makes the robot fall short
  of the displacement the policy assumed, which is the one thing that breaks the
  correspondence with training. The node warns if `max_v` would clip a
  full-forward command.
- A slower rate does **not** give the policy more decisions per metre. Decisions
  per metre is fixed at `1 / (action * step_size_m)`. What it buys is more time
  for the robot to reach the commanded velocity, and fresher camera and odometry
  at each decision.

Start at 2 Hz, then verify the robot is doing what it was told:

```bash
python3 - <<'PYEOF'
import csv, numpy as np
r  = list(csv.DictReader(open("policy_1.csv")))
t  = np.array([float(a["t"]) for a in r])
xy = np.array([[float(a["x"]), float(a["y"])] for a in r])
v  = np.array([float(a["v"]) for a in r])
dt = np.diff(t)
commanded = v[:-1] * dt
actual    = np.linalg.norm(np.diff(xy, axis=0), axis=1)
print("per decision: commanded %.3f m, actual %.3f m, ratio %.2f"
      % (commanded.mean(), actual.mean(), actual.mean() / max(commanded.mean(), 1e-6)))
PYEOF
```

A ratio near 1.0 means the robot is tracking the commanded velocity. Much below
1.0 means it is not reaching it, so lower `--rate` until it does. Go slower in
tight spaces regardless; nothing about the policy requires 2 Hz.

### Making the robot behave like it did in simulation

Three settings, in order of how much they matter:

1. **Do not clip.** Keep `--max_v >= step_size_m * rate` and
   `--max_w >= yaw_step_rad * rate`, or a full-command action silently becomes
   a partial step and the geometry stops matching. The node warns about this.
2. **`--smooth 1.0`.** Simulation applies each action fully and instantly.
   Smoothing spreads it over two or three decisions, which the policy never
   experienced. If that is too abrupt, lower `--rate`, not `--smooth`.
3. **Verify tracking** with the log check below. A ratio near 1.0 means each
   decision covered the distance the policy assumed.

Faithful configuration, for a 0.25 m / 0.3 rad policy at 2 Hz:

```bash
--rate 2 --smooth 1.0 --max_v 0.6 --max_w 0.8 \
--timeout_s 60 --no_progress_s 15 --log policy_1.csv
```

**The one difference you cannot remove**, and do not need to: simulation turns
first and then drives straight, while the robot gets forward and yaw at the
same time and so drives an arc. For a full step (0.3 rad, 0.25 m) the arc has
radius 0.83 m, a chord of 0.249 m against 0.25 m, and an identical final
heading; the landing point is about 4 cm off to the side. Every decision
re-observes and corrects, so this is not worth engineering around.

`--smooth` is separate. Training applies each action fully and instantly, so
`--smooth 1.0` is the faithful setting and anything lower adds a lag the policy
never experienced. Against that, the real robot has inertia and raw commands at
2 Hz look abrupt. 0.7 is a reasonable compromise; use 1.0 with `--rate 1.5` if
the motion looks violent.

---

## 8. Troubleshooting

- `ModuleNotFoundError: numpy._core...` — the PYTHONPATH export was not run in
  this terminal, so conda's numpy 1 is shadowing the venv.
- No camera topic — section 2 step 3. No `/Odometry` — FAST-LIO needs the LiDAR
  running first.
- Robot ignores commands — motion_control is not running, or the robot is not in
  terrain mode.
- First inference takes 300-500 ms — CUDA warm-up. The median settles near 8 ms
  and the node prints it.
- `[CLI default]` in the startup lines — `env_config.json` is missing beside the
  checkpoint.
- Node stuck on "waiting for camera/odom" — one of the two topics is not
  publishing; check with `ros2 topic hz`.

---

## 9. Worked example: ppo_322704 (robot test 2026-09-09)

Verified by loading the checkpoint and running the node end to end with ROS
stubbed: it loads with stock stable-baselines3 (no custom classes), a 640x480
camera frame preprocesses to exactly the 336x224 it expects, and inference is
about 6 ms.

Facts from the checkpoint and its config:

| | |
|---|---|
| observation | `rgb (3, 224, 336)` channels-first, `goal (3,)` |
| action | `Box(-1, 1, (2,))` |
| step_size_m | 0.25 |
| yaw_step_rad | 0.3 |
| goal_radius | 1.0 |
| static_scene | True (trained on fused-static renders) |
| label_remap | 12:0 (person ignored in the reward) |

Copy to Thor:

```bash
ssh soar@<thor> 'mkdir -p ~/nav_policy/ppo322704/checkpoints'
scp ~/Downloads/ppo_322704_steps.zip soar@<thor>:~/nav_policy/ppo322704/checkpoints/
scp ~/Downloads/env_config.json      soar@<thor>:~/nav_policy/ppo322704/
scp scripts/deploy_go2.py            soar@<thor>:~/nav_policy/
```

Run:

```bash
~/nav_env/bin/python3 deploy_go2.py \
    --checkpoint ppo322704/checkpoints/ppo_322704_steps.zip \
    --goal_dx 4.0 --goal_dy 0.0 \
    --rate 2 --smooth 1.0 --max_v 0.6 --max_w 0.8 \
    --timeout_s 60 --no_progress_s 15 --log policy_1.csv
```

The startup lines must read exactly:

```
[deploy] observation from checkpoint: 336x224 (WxH)
[deploy] step 0.25 m, yaw 0.3 rad per decision  [.../env_config.json]
[deploy] at 2.0 Hz -> max v 0.50 m/s, max w 0.60 rad/s (clipped to 0.6/0.8)
[deploy] stop radius 1.00 m
```

Anything different means something did not travel with the checkpoint. In
particular `[CLI default]` means `env_config.json` is not beside it.

**What is known and unknown going in.** The policy was trained on generated
imagery that is measurably softer than a real camera feed, so its response to
sharp RealSense frames is untested. Its simulated corner failures were caused
by phantom obstacles in generated frames, which cannot occur on a real camera,
so those specific loops should not reappear; the corner behaviour is genuinely
unmeasured rather than known-bad. The simulated corner evaluation was
effectively three distinct trajectories, not ten.
