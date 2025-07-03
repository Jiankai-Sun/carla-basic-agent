from __future__ import annotations
import argparse, queue, sys
from pathlib import Path
from math import atan2, degrees

import numpy as np
import carla, cv2

def speed_mps(v: carla.Vector3D) -> float:
    return (v.x**2 + v.y**2 + v.z**2) ** 0.5

def yaw_wrap(deg_val: float) -> float:
    return (deg_val + 180.0) % 360.0 - 180.0

def init_video(path:str, size:tuple[int,int], fps:int):
    for codec in ("avc1","mp4v","MJPG"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if vw.isOpened():
            print(f"[INFO] Recording → {path} (codec {codec})")
            return vw
    print("[WARN] Could not open an MP4 codec – video disabled")
    return None

def main():
    ap = argparse.ArgumentParser("Autopilot Δ-logger + video")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--map",  default="Town05")
    ap.add_argument("--fps",  type=int, default=10)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--outfile", default="deltas.npy")
    ap.add_argument("--video",   default="autopilot.mp4")
    args = ap.parse_args()

    dt       = 1.0 / args.fps
    n_ticks  = int(args.duration * args.fps)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world  = client.load_world(args.map)

    settings = world.get_settings()
    settings.synchronous_mode     = True
    settings.fixed_delta_seconds  = dt
    world.apply_settings(settings)

    tm = client.get_trafficmanager(args.tm_port); tm.set_synchronous_mode(True)

    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.filter(args.vehicle_filter)[0]
    spawn_tf = world.get_map().get_spawn_points()[0]
    ego = world.spawn_actor(ego_bp, spawn_tf)
    ego.set_autopilot(True, args.tm_port)
    print("[INFO] Ego on autopilot – logging deltas and video …")

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "100")
    cam_tf = carla.Transform(carla.Location(x=1.6, z=1.4))
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    img_q: "queue.Queue[carla.Image]" = queue.Queue(2)
    cam.listen(lambda img: img_q.put(img) if not img_q.full() else None)

    video = init_video(args.video, (1280, 720), args.fps)

    deltas = []
    prev_speed = speed_mps(ego.get_velocity())
    prev_yaw   = yaw_wrap(ego.get_transform().rotation.yaw)

    for k in range(n_ticks):
        world.tick()

        cur_speed = speed_mps(ego.get_velocity())
        cur_yaw   = yaw_wrap(ego.get_transform().rotation.yaw)

        delta_v   = cur_speed - prev_speed
        delta_yaw = yaw_wrap(cur_yaw - prev_yaw)
        
        # print(delta_yaw)

        deltas.append([delta_v, delta_yaw])
        prev_speed, prev_yaw = cur_speed, cur_yaw

        try:
            img = img_q.get_nowait()
            if video:
                arr = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
                video.write(cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR))
        except queue.Empty:
            pass

        if (k+1) % (args.fps*5) == 0:
            print(f"  tick {k+1:4d}/{n_ticks}  Δv={delta_v:+5.3f}  Δψ={delta_yaw:+6.2f}")

    deltas_np = np.asarray(deltas, dtype=np.float32)
    np.save(Path(args.outfile).expanduser(), deltas_np)
    print(f"[INFO] Saved {len(deltas_np)} samples → {args.outfile}")
    if video:
        video.release()
        print(f"[INFO] Video saved → {args.video}")

    cam.stop(); cam.destroy(); ego.destroy()
    tm.set_synchronous_mode(False)
    settings.synchronous_mode = False; world.apply_settings(settings)

if __name__ == "__main__":
    main()
