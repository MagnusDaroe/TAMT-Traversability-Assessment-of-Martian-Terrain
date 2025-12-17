import omni.usd
import omni.kit.commands
from pxr import UsdGeom, Gf
import random, os, math

usd_context = omni.usd.get_context()
stage = usd_context.get_stage()
random.seed()

# --- Paths (avoid spaces) ---
asset_dir = "DataGenerationForTerrainEstimation/MarsAssets/Ground_RG_Map_2"

# --- Gather assets ---
terrain_assets = [
    os.path.join(asset_dir, f)
    for f in os.listdir(asset_dir)
    if f.endswith(".usd")
]
if not terrain_assets:
    print("No USD files found in:", asset_dir)
else:
    print(f"Found {len(terrain_assets)} terrain assets.")

# =========================
#  Crater spawn limiting
# =========================
CRATER_NAMES = {"crater_ground_1", "crater_ground_2", "crater_ground_3"}
MAX_CRATER_COUNT = 5
crater_spawned = 0  # global counter

non_crater_assets = []
for a in terrain_assets:
    base = os.path.splitext(os.path.basename(a))[0].lower()
    if base not in CRATER_NAMES:
        non_crater_assets.append(a)

def pick_asset_with_crater_limit():
    """Pick an asset while enforcing a global limit for crater assets."""
    global crater_spawned
    if crater_spawned >= MAX_CRATER_COUNT and non_crater_assets:
        return random.choice(non_crater_assets)

    for _ in range(100):
        asset = random.choice(terrain_assets)
        base = os.path.splitext(os.path.basename(asset))[0].lower()
        if base in CRATER_NAMES:
            if crater_spawned >= MAX_CRATER_COUNT:
                continue
            crater_spawned += 1
        return asset

    # Fallbacks
    if non_crater_assets:
        return random.choice(non_crater_assets)
    return random.choice(terrain_assets)

# --- Root prim ---
terrain_root_path = "/World/Terrain"
if stage.GetPrimAtPath(terrain_root_path):
    omni.kit.commands.execute("DeletePrims", paths=[terrain_root_path])
omni.kit.commands.execute("CreatePrim", prim_type="Xform", prim_path=terrain_root_path)

# =========================
#  Grid placement settings
# =========================
# Domain
x_range = (-8, 8)
y_range = (-8, 8)

# Grid spacing (meters). The number of placements will be
#   N = len(X) * len(Y)
GRID_SPACING = 0.7

# Optional tiny random jitter to avoid perfectly regular patterns (set to 0.0 for none)
JITTER = 0.0  # e.g., 0.05

# Transform randomization
yaw_limit_deg   = 0           # yaw about Z
scale_min, scale_max = 0.8, 1.2

# Build grid coordinates (inclusive of end when divisible by spacing)
def _frange(lo, hi, step):
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1
    return [lo + i * step for i in range(n)]

xs = _frange(x_range[0], x_range[1], GRID_SPACING)
ys = _frange(y_range[0], y_range[1], GRID_SPACING)

# Create list of grid points and (optionally) shuffle the visiting order
grid_points = [(x, y) for y in ys for x in xs]
random.shuffle(grid_points)  # comment out if you prefer scanline order

# Limit by grid size (you can also slice if you want fewer than grid points)
num_to_place = len(grid_points)

# For simple bookkeeping / stats
placed_count = 0
per_asset_counts = {}

# ============================================================
#  Placement loop (grid-based)
# ============================================================
for i, (x, y) in enumerate(grid_points[:num_to_place]):
    asset = pick_asset_with_crater_limit()

    # Random transform (no tilt, yaw only; yaw_limit_deg=0 -> no rotation)
    yaw_z = random.uniform(-yaw_limit_deg, yaw_limit_deg)
    scale = random.uniform(scale_min, scale_max)

    # Optional small jitter
    jx = random.uniform(-JITTER, JITTER) if JITTER > 0 else 0.0
    jy = random.uniform(-JITTER, JITTER) if JITTER > 0 else 0.0

    prim_path = f"{terrain_root_path}/terrain_{i:05d}"

    # Reference the asset directly at prim_path
    omni.kit.commands.execute(
        "CreateReferenceCommand",
        usd_context=usd_context,
        path_to=prim_path,
        asset_path=asset,
    )

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)

    # --- Translate in X–Y plane, flat on Z=0 ---
    translate_ops = [op for op in xform.GetOrderedXformOps() if "translate" in op.GetOpName()]
    (translate_ops[0] if translate_ops else xform.AddTranslateOp()).Set(Gf.Vec3f(x + jx, y + jy, 0.0))

    # --- Rotation (yaw about Z) ---
    rotate_ops = [op for op in xform.GetOrderedXformOps() if "rotate" in op.GetOpName()]
    if rotate_ops and rotate_ops[0].GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
        rotate_ops[0].Set(Gf.Vec3d(0.0, 0.0, yaw_z))
        r_op = rotate_ops[0]
    else:
        r_op = xform.AddRotateXYZOp()
        r_op.Set(Gf.Vec3d(0.0, 0.0, yaw_z))

    # --- Uniform scale ---
    scale_ops = [op for op in xform.GetOrderedXformOps() if "scale" in op.GetOpName()]
    (scale_ops[0] if scale_ops else xform.AddScaleOp()).Set(Gf.Vec3f(scale, scale, scale))

    # (Optional) enforce a consistent op order: scale -> rotate -> translate
    # This helps avoid the xform order warnings.
    ops = xform.GetOrderedXformOps()
    t_op = next((op for op in ops if "translate" in op.GetOpName()), None)
    s_op = next((op for op in ops if "scale" in op.GetOpName()), None)
    if t_op and s_op and r_op:
        xform.SetXformOpOrder([s_op, r_op, t_op])

    # Stats
    base = os.path.basename(asset)
    per_asset_counts[base] = per_asset_counts.get(base, 0) + 1
    placed_count += 1

print(f"Generated {placed_count} terrain patches on a {len(xs)} x {len(ys)} grid "
      f"(spacing={GRID_SPACING} m, jitter={JITTER}).")
print(f"   Craters spawned (Crater_Ground_1/2/3): {crater_spawned}/{MAX_CRATER_COUNT}")

print("\nAsset placement counts:")
for k in sorted(per_asset_counts.keys()):
    print(f"{k}: {per_asset_counts[k]}")
