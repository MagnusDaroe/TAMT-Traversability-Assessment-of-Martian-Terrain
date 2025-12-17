import omni.usd
import omni.kit.commands
import omni.replicator.core as rep
from pxr import UsdGeom, Gf, Sdf
import omni.timeline, omni.kit.app
import asyncio, random, os, datetime
import numpy as np
from PIL import Image

# ==============================================================
# CONFIGURATION
# ==============================================================
ROVER_ASSET = "/home/tamt/Desktop/IsaacSimProjects/DataGenerationForTerrainEstimation/Rover/Leo_rover_control_lidar_camera3.usdz"
ROVER_PATH = "/World/Rover"
CAMERA_PATH = "/World/Rover/Main_Body/Base_Body/ZED_X/base_link/ZED_X/CameraLeft"

AREA_RANGE = (-5, 5)
SPAWN_HEIGHT = 0.5
WAIT_TIME_SEC = 5.0
RESOLUTION = (512, 512)
OUT_ROOT = os.path.expanduser("~/replicator_output/rover_snapshots")

# ==============================================================
# STAGE / CONTEXT
# ==============================================================
ctx = omni.usd.get_context()
stage = ctx.get_stage()
app = omni.kit.app.get_app()
timeline = omni.timeline.get_timeline_interface()

# ==============================================================
# 1.  SPAWN ROVER (clean payload spawn)
# ==============================================================
# Delete old rover safely
if stage.GetPrimAtPath(ROVER_PATH):
    omni.kit.commands.execute("DeletePrims", paths=[ROVER_PATH])
    asyncio.get_event_loop().run_until_complete(app.next_update_async())

# Create Xform prim
omni.kit.commands.execute("CreatePrim", prim_type="Xform", prim_path=ROVER_PATH)

# Add payload correctly using Sdf.Payload
payload = Sdf.Payload(ROVER_ASSET)
omni.kit.commands.execute(
    "AddPayload",
    stage=stage,
    prim_path=ROVER_PATH,
    payload=payload,
)

print(f"Spawned rover payload from: {ROVER_ASSET}")

# Randomize spawn position & yaw
x, y, z = random.uniform(*AREA_RANGE), random.uniform(*AREA_RANGE), SPAWN_HEIGHT
yaw = random.uniform(0, 360)

rover_prim = stage.GetPrimAtPath(ROVER_PATH)
xform = UsdGeom.Xformable(rover_prim)
ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
(ops.get("xformOp:translate") or xform.AddTranslateOp()).Set(Gf.Vec3f(x, y, z))
(ops.get("xformOp:rotateXYZ") or xform.AddRotateXYZOp()).Set(Gf.Vec3f(0, 0, yaw))
print(f"Position ({x:.2f}, {y:.2f}, {z:.2f}), yaw={yaw:.1f}")

# ==============================================================
# 2.  ASYNC CAPTURE SEQUENCE
# ==============================================================
async def main_sequence():
    # Wait for camera prim to exist
    print(f"Waiting for camera prim: {CAMERA_PATH}")
    for _ in range(400):
        prim = stage.GetPrimAtPath(CAMERA_PATH)
        if prim and prim.IsValid():
            print("Camera prim found and valid!")
            break
        await app.next_update_async()
    else:
        print("Camera not found, aborting.")
        return

    # --- Replicator setup ---
    rep.orchestrator.set_capture_on_play(True)
    rep.orchestrator.run()

    rp = rep.create.render_product(CAMERA_PATH, RESOLUTION)
    annotators = {
        "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
        "semantic": rep.AnnotatorRegistry.get_annotator("semantic_segmentation"),
        "instance": rep.AnnotatorRegistry.get_annotator("instance_segmentation"),
    }
    for a in annotators.values():
        a.attach(rp)

    # --- Start simulation ---
    if not timeline.is_playing():
        timeline.play()

    print(f"Waiting {WAIT_TIME_SEC}s before capture...")
    t0 = app.get_time().elapsed_seconds
    while app.get_time().elapsed_seconds - t0 < WAIT_TIME_SEC:
        await app.next_update_async()

    print("Capturing frame...")
    await rep.orchestrator.step_async(rt_subframes=1, pause_timeline=False)
    await app.next_update_async()

    # --- Save results ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"session_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    for name, annot in annotators.items():
        data = annot.get_data()
        if not data:
            print(f"No data for {name}")
            continue
        arr = np.asarray(data["data"] if isinstance(data, dict) else data)
        out_path = os.path.join(out_dir, f"{name}.png")
        if arr.ndim == 3:
            Image.fromarray(arr[..., :3].astype("uint8")).save(out_path)
        elif arr.ndim == 2:
            Image.fromarray(arr.astype("uint16")).save(out_path)
        print(f"Saved {name} → {out_path}")

    print(f"Capture complete! Results in {out_dir}")

# Run the async capture sequence
asyncio.ensure_future(main_sequence())

