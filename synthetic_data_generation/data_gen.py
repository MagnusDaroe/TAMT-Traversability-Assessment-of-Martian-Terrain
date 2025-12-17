import os, json, datetime, asyncio, random, math, csv
import numpy as np
from PIL import Image

import omni.usd, omni.kit.app
import omni.replicator.core as rep
from omni.timeline import get_timeline_interface
from pxr import UsdGeom, Sdf, Gf

# Try OpenCV for polygon extraction (YOLO seg). If unavailable, we skip .txt polygons gracefully.
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# =================== POSTPROCESSING ON DATA ===================

ENABLE_POST_SPLIT_AUG = True
EXTERNAL_SCRIPT_PATH = "DataGenerationForTerrainEstimation/PythonScripts/texture_editing.py"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TRAIN_MULTIPLIER = 2.0
AUGMENT_TRAIN = True

# =================== CONFIG ===================

CAMERA_RIG_PATH = "/World/RandomCamRig"
CAMERA_PATH = f"{CAMERA_RIG_PATH}/Camera"

RESOLUTION = (960, 540)
AREA_RANGE = (-5.0, 5.0)
FIXED_Z = 0.8
Z_MIN_CLAMP = 0.05
NUM_CAPTURES = 30
OUT_ROOT = os.path.expanduser("~/replicator_output/rover_data")

CAPTURE_RGB = True
CAPTURE_SEMANTIC = True
CAPTURE_YOLO_TXT = True
CAPTURE_DEPTH = True
CAPTURE_NORMALS = True
CAPTURE_POINTCLOUD = True
SAVE_ROVER_POSES = True

ENABLE_LOCAL_JITTER = False  # set True to enable local jitter

VERBOSE_PLACEMENT = False
PROGRESS_EVERY_N_FRAMES = 50

USE_MANUAL_POSE = False
MANUAL_POSITION_XYZ = (0.0, 0.0, 0.80)
MANUAL_RPY_DEG = (0.0, 0.0, 0.0)

ROLL_RANGE_DEG = (0.0, 0.0)
PITCH_RANGE_DEG = (20.0, 20.0)  # Leorover camera is tilted 20 degrees downwards
YAW_RANGE_DEG = (0.0, 360.0)

LOCAL_JITTER_M = {
    "right": (-0.30, 0.30),
    "up": (0.00, 0.00),
    "forward": (-0.50, 0.50),
}

USE_PATH_PLACEMENT = True
PATH_START_XYZ = (0.0, 0.0, FIXED_Z)
PATH_END_XYZ   = (0.0, 0.0, FIXED_Z)
PATH_NUM_SAMPLES = 30

# Desired ZED-like FOV (degrees)
HFOV_DEG = 110.0   # horizontal FOV
VFOV_DEG = 70.0    # vertical FOV

FOCAL_MM = 2.1  # Used to be 2.208

TEXTURE_SETTLE_FRAMES = 5

MIN_COMPONENT_AREA_PX = 20
MIN_POLY_POINTS = 6

SEM_ATTR = "semantics:class"

FIXED_CLASSES = ["soil", "bedrock", "sand", "rocks", "hole"]

FIXED_CLASS_COLORS = {
    "soil": [0, 90, 255],
    "bedrock": [0, 200, 0],
    "sand": [255, 255, 0],
    "rocks": [255, 128, 0],
    "hole": [255, 0, 0],
}

NAME_ALIASES = {
    "loose sand": "sand",
    "rock": "rocks",
    "big rock": "rocks",
    "rocks": "rocks",
    "bedrock": "bedrock",
    "soil": "soil",
    "hole": "hole",
    "holes": "hole",
}

# =============================================

app = omni.kit.app.get_app()
timeline = get_timeline_interface()
stage = omni.usd.get_context().get_stage()


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


async def _tick(n=1):
    for _ in range(n):
        await app.next_update_async()

# ------------------------------ helpers ---------------------------


def _hsv_to_rgb(h, s, v):
    i = int(h * 6)
    f = h * 6 - i
    p = int(255 * v * (1 - s))
    q = int(255 * v * (1 - f * s))
    t = int(255 * v * (1 - (1 - f) * s))
    v255 = int(255 * v)
    i %= 6
    return [
        (v255, t, p), (q, v255, p), (p, v255, t),
        (p, q, v255), (t, p, v255), (v255, p, q)
    ][i]


def _fallback_color_for_label(lbl):
    h = ((hash(lbl or "") & 0xFFFFFFFF) * 2654435761 % 360) / 360.0
    return _hsv_to_rgb(h, 0.7, 1.0)


def _colorize_fixed_class_map(class_map, idx_to_color):
    H, W = class_map.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    uniq = np.unique(class_map)
    for u in uniq:
        if u >= 0:
            out[class_map == u] = np.array(idx_to_color.get(int(u), [128, 128, 128]), dtype=np.uint8)
    return out

# ---------- Rover to Camera static transform ----------

ROVER_TO_CAM_TRANSLATION = np.array([0.157499, 0.059899, 0.238857], dtype=float)
ROVER_TO_CAM_QUAT_XYZW = np.array([0.5, -0.5, -0.5, 0.5], dtype=float)  # [x,y,z,w]


def _quat_xyzw_to_rot(q):
    """Quaternion [x,y,z,w] → 3×3 rotation matrix (Hamilton)."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) + 1e-12
    x /= n
    y /= n
    z /= n
    w /= n

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),        2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz),  2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),        1.0 - 2.0 * (xx + yy)],
    ], dtype=float)


def _make_T(R, t):
    """Build 4×4 transform from R (3×3) and t (3,)."""
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def _invert_T(R, t):
    """Invert a rigid transform given by R, t. Returns (R_inv, t_inv)."""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


# Precompute rover→camera rotation/translation (T_CR)
R_CR = _quat_xyzw_to_rot(ROVER_TO_CAM_QUAT_XYZW)  # rover to camera rotation
t_CR = ROVER_TO_CAM_TRANSLATION                   # camera origin in rover frame

# ----------------------- semantics mirroring -------------------------

def _get_own_label(prim):
    a = prim.GetAttribute(SEM_ATTR)
    if a and a.HasAuthoredValue():
        try:
            return a.Get()
        except Exception:
            return None
    return None


def _nearest_ancestor_label(prim):
    p = prim
    while p and p.GetPath() != Sdf.Path.absoluteRootPath:
        lbl = _get_own_label(p)
        if lbl:
            return lbl
        p = p.GetParent()
    return None


def mirror_labels_to_meshes_session(stage, root="/World"):
    added = already = total = 0
    for prim in stage.TraverseAll():
        if prim.GetTypeName() != "Mesh":
            continue
        if not prim.GetPath().pathString.startswith(root):
            continue

        total += 1
        if _get_own_label(prim):
            already += 1
            continue

        lbl = _nearest_ancestor_label(prim)
        if not lbl:
            continue

        prim.CreateAttribute(SEM_ATTR, Sdf.ValueTypeNames.String).Set(lbl)
        added += 1

    print(f"[INFO] Semantic mirroring complete: added={added}, existed={already}, total_meshes={total}")

# ------------------------------ xform-op utilities ---------------------------


def _get_or_add_translate_op(xf: UsdGeom.Xformable):
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            return op
    return xf.AddTranslateOp()


def _get_or_add_rotateXYZ_op(xf: UsdGeom.Xformable):
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:rotateXYZ":
            return op
    return xf.AddRotateXYZOp()


def _set_order_safe(xf: UsdGeom.Xformable, desired_first=None):
    ops = list(xf.GetOrderedXformOps())
    if desired_first is not None:
        ops = [desired_first] + [op for op in ops if op != desired_first]
    xf.SetXformOpOrder(ops)

# ------------------------------ camera rig creation --------------------------


def create_or_reset_camera_rig(stage):
    # 1. Create or fetch the rig Xform
    rig_prim = stage.GetPrimAtPath(CAMERA_RIG_PATH)
    if not rig_prim:
        omni.kit.commands.execute("CreatePrim", prim_type="Xform", prim_path=CAMERA_RIG_PATH)
        rig_prim = stage.GetPrimAtPath(CAMERA_RIG_PATH)

    rig_xf = UsdGeom.Xformable(rig_prim)

    # Rig translate op
    rig_t = _get_or_add_translate_op(rig_xf)

    # Rig rotation op
    rig_r = _get_or_add_rotateXYZ_op(rig_xf)

    # Ensure translate occurs first
    _set_order_safe(rig_xf, desired_first=rig_t)

    # ---------- compute apertures for the requested FOV ----------
    hfov = math.radians(HFOV_DEG)
    vfov = math.radians(VFOV_DEG)

    h_ap = 2.0 * FOCAL_MM * math.tan(hfov / 2.0)   # mm
    v_ap = 2.0 * FOCAL_MM * math.tan(vfov / 2.0)   # mm

    # 2. Create or fetch camera prim and Xform
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not cam_prim:
        omni.kit.commands.execute(
            "CreatePrim",
            prim_type="Camera",
            prim_path=CAMERA_PATH,
            attributes={
                "projection": "perspective",
                "focalLength": FOCAL_MM,          # mm
                "horizontalAperture": h_ap,       # mm
                "verticalAperture": v_ap,         # mm
                "clippingRange": Gf.Vec2f(0.01, 5000.0),
            },
            select_new_prim=False,
        )
        cam_prim = stage.GetPrimAtPath(CAMERA_PATH)

    cam_xf = UsdGeom.Xformable(cam_prim)

    # Make sure camera uses these intrinsics
    cam = UsdGeom.Camera(cam_prim)
    cam.GetFocalLengthAttr().Set(FOCAL_MM)
    cam.GetHorizontalApertureAttr().Set(h_ap)
    cam.GetVerticalApertureAttr().Set(v_ap)

    # 3. Remove previous ops from camera (reset transform)
    try:
        existing_ops = cam_xf.GetOrderedXformOps()
        for op in existing_ops:
            cam_xf.RemoveXformOp(op)
    except Exception:
        cam_prim.RemoveProperty("xformOpOrder")

    # This ensures the camera looks along +X in world coordinates.
    fix_op = cam_xf.AddRotateXYZOp()
    fix_op.Set(Gf.Vec3f(90.0, 0.0, -90.0))

    # Set op order to contain only this correction
    cam_xf.SetXformOpOrder([fix_op])

    return rig_t, rig_r

# -------------------------- world transform extraction -----------------------


def _camera_world_rotation_matrix():
    rig = stage.GetPrimAtPath(CAMERA_RIG_PATH)
    T = UsdGeom.XformCache().GetLocalToWorldTransform(rig)

    R = np.array([
        [T[0][0], T[0][1], T[0][2]],
        [T[1][0], T[1][1], T[1][2]],
        [T[2][0], T[2][1], T[2][2]],
    ], dtype=float)

    return R


def _get_world_R_t():
    rig = stage.GetPrimAtPath(CAMERA_RIG_PATH)
    T = UsdGeom.XformCache().GetLocalToWorldTransform(rig)

    R = np.array([
        [T[0][0], T[0][1], T[0][2]],
        [T[1][0], T[1][1], T[1][2]],
        [T[2][0], T[2][1], T[2][2]],
    ], dtype=float)

    t = np.array([T[3][0], T[3][1], T[3][2]], dtype=float)

    return R, t, T


def _rot_to_quat_xyzw(R: np.ndarray):
    q = np.empty(4)
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q[3] = 0.25 * S
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = (R[0, 2] - R[2, 0]) / S
        q[2] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[3] = (R[2, 1] - R[1, 2]) / S
        q[0] = 0.25 * S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[3] = (R[0, 2] - R[2, 0]) / S
        q[0] = (R[0, 1] + R[1, 0]) / S
        q[1] = 0.25 * S
        q[2] = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[3] = (R[1, 0] - R[0, 1]) / S
        q[0] = (R[0, 2] + R[2, 0]) / S
        q[1] = (R[1, 2] + R[2, 1]) / S
        q[2] = 0.25 * S

    q /= np.linalg.norm(q) + 1e-12
    return q

# ------------------------------ clamp helpers ------------------------------


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _clamp_xy_to_area(x, y):
    lo, hi = AREA_RANGE
    return _clamp(x, lo, hi), _clamp(y, lo, hi)


# ----------------------- PATH PLACEMENT HELPERS -----------------------------

def _generate_linear_path(start_xyz, end_xyz, num_samples):
    """Generate num_samples positions linearly between start and end."""
    sx, sy, sz = start_xyz
    ex, ey, ez = end_xyz
    if num_samples <= 1:
        return [(sx, sy, sz)]
    positions = []
    for k in range(num_samples):
        alpha = float(k) / float(num_samples)
        x = sx + (ex - sx) * alpha
        y = sy + (ey - sy) * alpha
        z = sz + (ez - sz) * alpha
        positions.append((x, y, z))
    return positions


def _compute_path_yaw_deg(start_xyz, end_xyz):
    """Yaw that aligns rover +X axis with the path direction in XY plane."""
    sx, sy, _ = start_xyz
    ex, ey, _ = end_xyz
    dx = ex - sx
    dy = ey - sy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


# ----------------------- camera placement (RANDOM) -------------------------


def place_camera_fixed_height_with_local_jitter(rig_t, rig_r):
    x = random.uniform(*AREA_RANGE)
    y = random.uniform(*AREA_RANGE)
    z = max(FIXED_Z, Z_MIN_CLAMP)
    rig_t.Set(Gf.Vec3f(x, y, z))

    roll = random.uniform(*ROLL_RANGE_DEG)
    pitch = random.uniform(*PITCH_RANGE_DEG)
    yaw = random.uniform(*YAW_RANGE_DEG)
    rig_r.Set(Gf.Vec3f(roll, pitch, yaw))

    # If jitter disabled, stop here
    if not ENABLE_LOCAL_JITTER:
        if VERBOSE_PLACEMENT:
            print(f"[INFO] Placement (no jitter) = ({x},{y},{z}), rpy=({roll},{pitch},{yaw})")
        return (x, y, z)

    # -------- local jitter (only if enabled) --------
    j_right = random.uniform(*LOCAL_JITTER_M["right"])
    j_up = random.uniform(*LOCAL_JITTER_M["up"])
    j_forward = random.uniform(*LOCAL_JITTER_M["forward"])

    v_local = np.array([j_right, j_up, -j_forward], dtype=np.float32)

    R = _camera_world_rotation_matrix()
    delta_world = (R @ v_local).astype(np.float32)

    x2 = x + float(delta_world[0])
    y2 = y + float(delta_world[1])
    x2, y2 = _clamp_xy_to_area(x2, y2)

    z2 = max(FIXED_Z, Z_MIN_CLAMP)

    rig_t.Set(Gf.Vec3f(x2, y2, z2))

    if VERBOSE_PLACEMENT:
        print(f"[INFO] Random placement base=({x},{y},{z}) final=({x2},{y2},{z2})")

    return (x2, y2, z2)

# ----------------------- camera placement (MANUAL) -------------------------


def place_camera_manual_world(rig_t, rig_r):
    x, y, z = MANUAL_POSITION_XYZ
    roll, pitch, yaw = MANUAL_RPY_DEG

    rig_t.Set(Gf.Vec3f(float(x), float(y), float(z)))
    rig_r.Set(Gf.Vec3f(float(roll), float(pitch), float(yaw)))

    if VERBOSE_PLACEMENT:
        print(f"[INFO] Manual camera pose set to pos=({x},{y},{z}), rpy=({roll},{pitch},{yaw})")

    return (x, y, z)

# ------------------------ fixed mapping helpers ----------------------------


def _normalize_name(name: str) -> str:
    return name.strip().lower() if name else ""


def _map_to_fixed_index(label_name: str):
    if not label_name:
        return None
    n = _normalize_name(label_name)
    n = NAME_ALIASES.get(n, n)
    try:
        return FIXED_CLASSES.index(n)
    except ValueError:
        return None

# ----------------------- YOLO segmentation helpers ------------------------


def _mask_to_yolo_segments(mask_bin, class_idx):
    if not _HAS_CV2:
        return []

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask_bin.shape[:2]
    segs = []

    for cnt in contours:
        if len(cnt) < 3:
            continue
        area = float(cv2.contourArea(cnt))
        if area < MIN_COMPONENT_AREA_PX:
            continue

        eps = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < MIN_POLY_POINTS:
            continue

        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= float(W)
        pts[:, 1] /= float(H)

        seg = [int(class_idx)]
        for x, y in pts:
            seg.extend([float(x), float(y)])
        segs.append(seg)

    return segs


def _write_yolo_txt_from_class_map(class_map, out_txt_path):
    if not _HAS_CV2:
        with open(out_txt_path, "w") as f:
            f.write("")
        print("[WARN] cv2 not found: YOLO polygons disabled")
        return 0

    H, W = class_map.shape
    seg_count = 0
    with open(out_txt_path, "w") as f:
        for c in sorted(int(x) for x in np.unique(class_map) if x >= 0):
            mask = (class_map == c).astype(np.uint8)
            segs = _mask_to_yolo_segments(mask, c)

            for seg in segs:
                f.write(" ".join(str(v) for v in seg) + "\n")
                seg_count += 1

    return seg_count

# =========================================================
# MAIN CAPTURE
# =========================================================


async def main():
    session_layer = stage.GetSessionLayer()
    stage.SetEditTarget(session_layer)

    if CAPTURE_SEMANTIC:
        mirror_labels_to_meshes_session(stage, "/World")

    if timeline.is_playing():
        timeline.stop()
    await _tick(1)

    rig_t, rig_r = create_or_reset_camera_rig(stage)
    await _tick(2)

    rp = rep.create.render_product(CAMERA_PATH, RESOLUTION)
    rep.orchestrator.set_capture_on_play(False)

    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb") if CAPTURE_RGB else None
    if rgb_annot:
        rgb_annot.attach(rp)

    sem_annot = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation",
        init_params={"semanticTypes": ["class"]}
    ) if CAPTURE_SEMANTIC else None
    if sem_annot:
        sem_annot.attach(rp)

    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane") if CAPTURE_DEPTH else None
    if depth_annot:
        depth_annot.attach(rp)

    norm_annot = rep.AnnotatorRegistry.get_annotator("normals") if CAPTURE_NORMALS else None
    if norm_annot:
        norm_annot.attach(rp)

    pc_annot = rep.AnnotatorRegistry.get_annotator(
        "pointcloud",
        init_params={"includeUnlabelled": True}
    ) if CAPTURE_POINTCLOUD else None
    if pc_annot:
        pc_annot.attach(rp)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_ROOT, f"session_{ts}")

    images_dir = os.path.join(run_dir, "images")
    labels_dir = os.path.join(run_dir, "labels")
    depth_dir = os.path.join(run_dir, "depth")
    normals_dir = os.path.join(run_dir, "normals")
    pc_dir = os.path.join(run_dir, "pointclouds")

    for d in (images_dir, labels_dir, depth_dir, normals_dir, pc_dir):
        _ensure_dir(d)

    print(f"[INFO] Capturing {NUM_CAPTURES} frames → {run_dir}")
    
    # ----------------------------------------------------------
    # Optional path-based placement initialization
    # ----------------------------------------------------------
    path_positions = None
    path_yaw_deg = 0.0
    
    if USE_PATH_PLACEMENT:
        path_positions = _generate_linear_path(
            PATH_START_XYZ, PATH_END_XYZ, PATH_NUM_SAMPLES
        )
        path_yaw_deg = _compute_path_yaw_deg(PATH_START_XYZ, PATH_END_XYZ)

        if len(path_positions) != NUM_CAPTURES:
            raise ValueError(
                f"NUM_CAPTURES ({NUM_CAPTURES}) must equal PATH_NUM_SAMPLES "
                f"({PATH_NUM_SAMPLES}) when USE_PATH_PLACEMENT=True."
            )

        print(f"[INFO] Path placement enabled: {PATH_START_XYZ} to {PATH_END_XYZ}, " f"{PATH_NUM_SAMPLES} samples, yaw={path_yaw_deg:.2f}°")


    id_to_labels_json_path = os.path.join(run_dir, "id_to_labels.json")
    data_yaml_path = os.path.join(run_dir, "data.yaml")
    rover_poses_csv_path = os.path.join(run_dir, "rover_poses.csv")

    # Write semantic id/color mapping
    if CAPTURE_SEMANTIC:
        enriched = {}
        for idx, name in enumerate(FIXED_CLASSES):
            enriched[str(idx)] = {
                "class": name,
                "color_rgb": list(FIXED_CLASS_COLORS.get(name, _fallback_color_for_label(name)))
            }

        with open(id_to_labels_json_path, "w") as f:
            json.dump(enriched, f, indent=2)

        with open(data_yaml_path, "w") as f:
            f.write(f"path: {run_dir}\n")
            f.write("train: images\n")
            f.write("val: images\n\n")
            f.write("names:\n")
            for idx, name in enumerate(FIXED_CLASSES):
                f.write(f"  {idx}: {name}\n")

    rover_poses = []

    if not timeline.is_playing():
        timeline.play()

    # ----------------- CAPTURE LOOP ---------------------

    for i in range(NUM_CAPTURES):

        if USE_PATH_PLACEMENT:
            pos = path_positions[i]
            rig_t.Set(Gf.Vec3f(*pos))
            rig_r.Set(Gf.Vec3f(float(ROLL_RANGE_DEG[0]), float(PITCH_RANGE_DEG[0]), float(path_yaw_deg)))
            
        elif USE_MANUAL_POSE:
            place_camera_manual_world(rig_t, rig_r)
        else:
            place_camera_fixed_height_with_local_jitter(rig_t, rig_r)

        # let textures settle
        for _ in range(TEXTURE_SETTLE_FRAMES):
            await rep.orchestrator.step_async(rt_subframes=1, pause_timeline=False)
            await _tick(1)

        base = f"rgb_{i:06d}"
        rgb_png_rel = os.path.join("images", f"{base}.png")
        rgb_png_path = os.path.join(images_dir, f"{base}.png")
        yolo_txt_path = os.path.join(labels_dir, f"{base}.txt")

        # RGB
        if CAPTURE_RGB and rgb_annot:
            rgb = rgb_annot.get_data()
            if isinstance(rgb, dict):
                rgb = rgb.get("data")
            if rgb is not None:
                Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8)).save(rgb_png_path)

        # SEM + YOLO
        if CAPTURE_SEMANTIC and sem_annot:
            sem = sem_annot.get_data()
            if isinstance(sem, dict):
                sem_img = sem.get("data")
                sem_info = sem.get("info", {})
            else:
                sem_img, sem_info = sem, {}

            if sem_img is not None:
                ids = np.asarray(sem_img).astype(np.uint32)
                Image.fromarray(ids.clip(0, 65535).astype(np.uint16)).save(
                    os.path.join(labels_dir, f"sem_ids_{i:06d}.png")
                )

                id_to_name = {}
                src = sem_info.get("idToLabels", {}) if isinstance(sem_info, dict) else {}
                for k, meta in src.items():
                    try:
                        kid = int(k)
                    except Exception:
                        continue
                    name = None
                    if isinstance(meta, dict):
                        name = meta.get("class") or meta.get("label") or meta.get("name")
                    if name:
                        id_to_name[kid] = _normalize_name(name)

                H, W = ids.shape
                class_map = np.full((H, W), -1, dtype=np.int32)

                for uid in np.unique(ids):
                    if uid == 0:
                        continue
                    src_name = id_to_name.get(int(uid))
                    fixed_idx = _map_to_fixed_index(src_name) if src_name else None
                    if fixed_idx is not None:
                        class_map[ids == uid] = fixed_idx

                idx_to_color = {
                    i_cls: FIXED_CLASS_COLORS.get(name, _fallback_color_for_label(name))
                    for i_cls, name in enumerate(FIXED_CLASSES)
                }

                color_img = _colorize_fixed_class_map(class_map, idx_to_color)
                Image.fromarray(color_img).save(
                    os.path.join(labels_dir, f"sem_color_{i:06d}.png")
                )

                if CAPTURE_YOLO_TXT:
                    _ = _write_yolo_txt_from_class_map(class_map, yolo_txt_path)

        # DEPTH
        if CAPTURE_DEPTH and depth_annot:
            d = depth_annot.get_data()
            if d is not None:
                depth = np.asarray(d).astype(np.float32)
                np.save(os.path.join(depth_dir, f"depth_{i:06d}.npy"), depth)

                vmax = float(np.nanmax(depth)) if np.isfinite(depth).any() else 1.0
                scale = 65535.0 / max(vmax, 1e-6)

                Image.fromarray(
                    (np.nan_to_num(depth, 0.0) * scale).clip(0, 65535).astype(np.uint16)
                ).save(os.path.join(depth_dir, f"depth_{i:06d}.png"))

        # NORMALS
        if CAPTURE_NORMALS and norm_annot:
            n = norm_annot.get_data()
            if n is not None:
                normals = np.asarray(n).astype(np.float32)
                np.save(os.path.join(normals_dir, f"normals_{i:06d}.npy"), normals)

                Image.fromarray(
                    ((normals * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
                ).save(os.path.join(normals_dir, f"normals_{i:06d}.png"))

        # POINTCLOUD
        if CAPTURE_POINTCLOUD and pc_annot:
            pc = pc_annot.get_data()
            if pc is not None and isinstance(pc, dict) and "data" in pc:
                pts = np.asarray(pc["data"]).astype(np.float32)
                np.save(os.path.join(pc_dir, f"pc_{i:06d}.npy"), pts)

        # ROVER POSE: pose of rover in world frame
        if SAVE_ROVER_POSES:
            # 1. Get the rig pose in world coordinates
            R_WRIG, t_WRIG, _ = _get_world_R_t()

            # 2. Static translation from rover to camera (rover origin to camera)
            t_rover_to_cam = np.array([0.157499, 0.059899, 0.238857], dtype=float)

            # 3. Compute rover origin in world coordinates
            t_WR = t_WRIG - t_rover_to_cam

            # 4. Rover orientation (yaw-only) from rig rotation
            rpy_deg = rig_r.Get()  # Gf.Vec3f(roll, pitch, yaw)
            yaw_deg = float(rpy_deg[2])
            yaw = math.radians(yaw_deg)

            cy, sy = math.cos(yaw), math.sin(yaw)
            R_WR_yaw = np.array([
                [cy, -sy, 0.0],
                [sy,  cy, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=float)

            q_xyzw = _rot_to_quat_xyzw(R_WR_yaw)

            # 5. Save to list
            rover_poses.append({
                "frame_index": i,
                "image": rgb_png_rel,
                "tx": float(t_WR[0]),
                "ty": float(t_WR[1]),
                "tz": float(t_WR[2]),
                "qx": float(q_xyzw[0]),
                "qy": float(q_xyzw[1]),
                "qz": float(q_xyzw[2]),
                "qw": float(q_xyzw[3]),
            })


        if (i + 1) % max(1, PROGRESS_EVERY_N_FRAMES) == 0 or (i + 1) == NUM_CAPTURES:
            print(f"[INFO] Captured {i+1}/{NUM_CAPTURES}")

    # ----------------- Write rover pose CSV ---------------------

    if SAVE_ROVER_POSES:
        with open(rover_poses_csv_path, "w", newline="") as f:
            f.write("# rover_T_world (rover to world, yaw-only orientation). Quaternion = [x,y,z,w] Hamilton\n")
            writer = csv.writer(f)
            writer.writerow(["frame_index", "image", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
            for p in rover_poses:
                writer.writerow([
                    p["frame_index"], p["image"],
                    p["tx"], p["ty"], p["tz"],
                    p["qx"], p["qy"], p["qz"], p["qw"]
                ])

        print(f"[INFO] Rover poses saved to: {rover_poses_csv_path}")

    # Cleanup
    print("[INFO] Cleaning up annotators...")

    for a in (rgb_annot, sem_annot, depth_annot, norm_annot, pc_annot):
        if a is not None:
            try:
                a.detach()
            except Exception:
                pass

    try:
        rep.destroy.render_product(rp)
    except Exception:
        pass

    if timeline.is_playing():
        timeline.stop()

    await _tick(2)

    # ------------------ External split/augment ------------------

    if ENABLE_POST_SPLIT_AUG:
        print("[INFO] Running external dataset split & augmentation script...")

        final_out = os.path.join(run_dir, "yolo_ready")
        _ensure_dir(final_out)

        cmd = (
            f"python3 {EXTERNAL_SCRIPT_PATH} "
            f"{run_dir} {final_out} "
            f"--train {TRAIN_RATIO} --valid {VAL_RATIO} --test {TEST_RATIO} "
            f"--multiplier {TRAIN_MULTIPLIER} "
        )

        if not AUGMENT_TRAIN:
            cmd += " --no-augment"

        print("[INFO] Executing external script:")
        print(" ", cmd)
        ret = os.system(cmd)

        if ret != 0:
            print(f"[ERROR] External script failed with exit code {ret}")
        else:
            print("[INFO] External dataset processing completed successfully.")
            print(f"[INFO] YOLO-ready dataset path: {final_out}")

    print("[INFO] Done.")

    if CAPTURE_YOLO_TXT and not _HAS_CV2:
        print("[WARN] OpenCV missing — YOLO .txt masks were empty.")


# schedule main
asyncio.ensure_future(main())
print("[INFO] Scheduled. If nothing starts, press Play once.")
