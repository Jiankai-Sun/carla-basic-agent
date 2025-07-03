from __future__ import annotations
import argparse, queue, sys
from pathlib import Path
from math import cos, sin, radians

import numpy as np
import carla, cv2

def init_video(path: str, size: tuple[int, int], fps: int):
    """Open a VideoWriter with a codec that exists on this system."""
    for codec in ("avc1", "mp4v", "MJPG"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if vw.isOpened():
            print(f"[INFO] Recording → {path} (codec {codec})")
            return vw
    print("[WARN] Video disabled – no codec opened")
    return None


def yaw_to_vec(yaw_deg: float) -> carla.Vector3D:
    """Ground-plane forward unit-vector from a yaw angle (deg)."""
    return carla.Vector3D(cos(radians(yaw_deg)), sin(radians(yaw_deg)), 0.0)

def main():
    ap = argparse.ArgumentParser("Δ-Waypoint driver")
    ap.add_argument("--deltas", default="deltas.npy",
                    help="N×2 NumPy file: [Δspeed (m/s), Δyaw (deg)] per tick")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--map",  default="Town05")
    ap.add_argument("--fps",  type=int, default=10)
    ap.add_argument("--video", default="delta_run.mp4")
    ap.add_argument("--show_wp", action="store_true",
                    help="draw target waypoint every tick")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world  = client.load_world(args.map)

    dt = 1.0 / args.fps
    settings              = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = dt
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    spawn_tf = world.get_map().get_spawn_points()[0]
    x0, y0, z0 = spawn_tf.location.x, spawn_tf.location.y, spawn_tf.location.z
    yaw0       = spawn_tf.rotation.yaw

    deltas_path = Path(args.deltas).expanduser()
    if not deltas_path.is_file():
        sys.exit(f"[ERROR] Cannot find {deltas_path}")

    deltas = np.load(deltas_path)
    if deltas.ndim != 2 or deltas.shape[1] != 2:
        sys.exit("[ERROR] Δ file must be N×2: [Δspeed, Δyaw]")

    speeds = np.clip(np.cumsum(deltas[:, 0]), 0.0, None)   # m/s, never < 0
    yaws   = np.cumsum(deltas[:, 1]) + yaw0                # deg

    poses  = [carla.Location(x0, y0, z0)]
    for v, yaw in zip(speeds, yaws):
        step = yaw_to_vec(yaw) * v * dt        # displacement for this tick
        poses.append(poses[-1] + carla.Location(step.x, step.y, 0.0))

    # =======================================================================
    #  SPAWN EGO & SENSORS
    # =======================================================================
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.filter("vehicle.*model3")[0]
    ego    = world.spawn_actor(ego_bp, spawn_tf)

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "100")
    cam_tf  = carla.Transform(carla.Location(x=1.6, z=1.4))
    cam     = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    img_q: "queue.Queue[carla.Image]" = queue.Queue(2)
    cam.listen(lambda img: img_q.put(img) if not img_q.full() else None)

    video = init_video(args.video, (1280, 720), args.fps)

    # =======================================================================
    #  BUILD WAYPOINT PLAN
    # =======================================================================
    mp   = world.get_map()
    plan = []
    prev_wp = None
    for loc in poses[1:]:
        wp = mp.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if prev_wp is None or wp.transform.location.distance(
                prev_wp.transform.location) > 0.2:     # dedupe
            plan.append((wp, 4))   # 4 == RoadOption.LANEFOLLOW
            prev_wp = wp

    from local_planner import LocalPlanner
    lp = LocalPlanner(
        ego,
        {"target_speed": np.max(speeds)*3.6,   # km/h
         "sampling_radius": 0.5},              # finer than default 2 m
        mp)
    lp.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)

    # ­­­­­­­–– driving loop ­­­­­­­––
    err_hist = []
    try:
        for k, (v_target, yaw) in enumerate(zip(speeds, yaws), start=1):
            lp.set_speed(v_target * 3.6)      # km/h
            ctrl = lp.run_step(debug=args.show_wp)
            ego.apply_control(ctrl)
            world.tick()

            err = ego.get_location().distance(poses[k])
            err_hist.append(err)
            try:
                img = img_q.get_nowait()
                if video:
                    arr = np.frombuffer(img.raw_data, np.uint8).reshape(
                        (img.height, img.width, 4))
                    video.write(cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR))
            except queue.Empty:
                pass

            if k % (args.fps*5) == 0:
                print(f"  t={k*dt:5.1f}s  v={v_target:4.1f}  err={err:5.2f} m")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    err_arr = np.asarray(err_hist)
    print("\n═════ L2 position error ═════")
    print(f" samples   : {len(err_arr)}")
    print(f" mean  [m] : {err_arr.mean():8.3f}")
    print(f" RMS   [m] : {np.sqrt((err_arr**2).mean()):8.3f}")
    print(f" max   [m] : {err_arr.max():8.3f}")

    if video:
        video.release()
        print(f"[INFO] Video saved → {args.video}")

    cam.stop(); cam.destroy(); ego.destroy()
    tm.set_synchronous_mode(False)
    settings.synchronous_mode = False
    world.apply_settings(settings)

if __name__ == "__main__":
    main()
