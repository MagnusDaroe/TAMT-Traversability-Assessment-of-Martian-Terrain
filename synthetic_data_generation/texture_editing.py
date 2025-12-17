import sys, os
import numpy as np
from PIL import Image

def tint_rgba(img_rgba,
              rgb_scale=(1.20, 0.93, 0.90),       # subtle warm shift
              blend_rgb=(0.502, 0.478, 0.290),    # ~ (128,122,74) RGB colours
              #blend_rgb=(0.71, 0.32, 0.22),       # ~ #b55239 Mars red
              blend_alpha=0.40):                  # 0..1 strength
    arr = np.asarray(img_rgba).astype(np.float32)  # HxWx4 RGBA
    rgb = arr[..., :3] / 255.0
    a   = arr[..., 3:4] / 255.0

    # 1) per-channel multiply
    rgb = np.clip(rgb * np.array(rgb_scale, dtype=np.float32), 0, 1)

    # 2) blend toward Mars red
    if blend_rgb is not None and blend_alpha > 0:
        target = np.array(blend_rgb, dtype=np.float32)
        rgb = np.clip((1 - blend_alpha) * rgb + blend_alpha * target, 0, 1)

    out = np.concatenate([(rgb * 255).round().astype(np.uint8),
                          (a   * 255).round().astype(np.uint8)], axis=-1)
    return Image.fromarray(out, mode="RGBA")

def save_side_by_side(before_rgb, after_rgb, out_path):
    w = min(before_rgb.width, after_rgb.width)
    h = min(before_rgb.height, after_rgb.height)
    before_rgb = before_rgb.resize((w, h), Image.LANCZOS)
    after_rgb  = after_rgb.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w*2, h), (0,0,0))
    canvas.paste(before_rgb, (0, 0))
    canvas.paste(after_rgb, (w, 0))
    canvas.save(out_path)

def main():
    if len(sys.argv) < 2:
        print("Usage: python mars_tint_min.py <image.png|jpg>")
        sys.exit(1)
    in_path = sys.argv[1]
    if not os.path.isfile(in_path):
        print(f"File not found: {in_path}")
        sys.exit(1)

    img = Image.open(in_path).convert("RGBA")
    out_img = tint_rgba(img)

    root, _ = os.path.splitext(in_path)
    out_png = root + "_yellow.png"
    out_img.save(out_png)
    print(f"Saved tinted → {out_png}")

    cmp_png = root + "_mars_comparison.png"
    save_side_by_side(img.convert("RGB"), out_img.convert("RGB"), cmp_png)
    print(f"Saved comparison → {cmp_png}")

if __name__ == "__main__":
    main()
