#!/usr/bin/env python3
import os
import argparse
import random
import shutil
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

# =====================================================================================
# Helpers
# =====================================================================================

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_image(path):
    return Image.open(path)

def save_image(img, path):
    _ensure_dir(os.path.dirname(path))
    img.save(path)

def copy_file(src, dst):
    _ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)

def yolo_label_path(rgb_path):
    """Convert images/rgb_xxx.png → labels/rgb_xxx.txt"""
    name = os.path.basename(rgb_path).replace(".png", ".txt")
    return os.path.join("labels", name)

# =====================================================================================
# Augmentation functions
# =====================================================================================

def aug_flip(img, seg):
    return img.transpose(Image.FLIP_LEFT_RIGHT), seg.transpose(Image.FLIP_LEFT_RIGHT)

def aug_rotate(img, seg):
    angle = random.uniform(-10, 10)
    return img.rotate(angle, resample=Image.BILINEAR), seg.rotate(angle, resample=Image.NEAREST)

def aug_brightness(img, seg):
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Brightness(img).enhance(factor), seg

def aug_contrast(img, seg):
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Contrast(img).enhance(factor), seg

def aug_color(img, seg):
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Color(img).enhance(factor), seg

def aug_blur(img, seg):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.0))), seg

def aug_noise(img, seg):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 5, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy), seg

AUG_FUNCS = [
    aug_flip,
    aug_rotate,
    aug_brightness,
    aug_contrast,
    aug_color,
    aug_blur,
    aug_noise,
]

def apply_random_augmentation(img, seg):
    """Apply 1–3 random augmentations."""
    n = random.randint(1, 3)
    funcs = random.sample(AUG_FUNCS, n)
    for f in funcs:
        img, seg = f(img, seg)
    return img, seg

# =====================================================================================
# Dataset split
# =====================================================================================

def split_indices(n, train_r, val_r, test_r):
    idxs = list(range(n))
    random.shuffle(idxs)
    n_train = int(train_r * n)
    n_val   = int(val_r   * n)
    train = idxs[:n_train]
    val   = idxs[n_train:n_train+n_val]
    test  = idxs[n_train+n_val:]
    return train, val, test

# =====================================================================================
# Processing
# =====================================================================================

def process_set(
    indices,
    all_rgb_files,
    src_root,
    out_root,
    subset_name,
    multiplier,
    augment_enabled,
):
    print(f"📁 Processing {subset_name} ({len(indices)} samples)...")

    for idx in indices:
        rgb_rel = all_rgb_files[idx]  # e.g. "images/rgb_000012.png"
        rgb_src = os.path.join(src_root, rgb_rel)
        name = os.path.splitext(os.path.basename(rgb_rel))[0]

        seg_color_src = os.path.join(src_root, "labels", f"sem_color_{name[4:]}.png")
        yolo_src      = os.path.join(src_root, "labels", f"{name}.txt")

        # Output paths
        rgb_out_dir = os.path.join(out_root, subset_name, "images")
        lbl_out_dir = os.path.join(out_root, subset_name, "labels")

        rgb_out = os.path.join(rgb_out_dir, os.path.basename(rgb_rel))
        seg_out = os.path.join(lbl_out_dir, f"{name}.png")
        yolo_out = os.path.join(lbl_out_dir, f"{name}.txt")

        # Copy main data
        copy_file(rgb_src, rgb_out)
        if os.path.exists(seg_color_src):
            copy_file(seg_color_src, seg_out)
        if os.path.exists(yolo_src):
            copy_file(yolo_src, yolo_out)

        # Augment (only for train set)
        if subset_name == "train" and augment_enabled:
            rgb_img = load_image(rgb_src)
            seg_img = load_image(seg_color_src) if os.path.exists(seg_color_src) else None

            for a in range(int(multiplier) - 1):  
                aug_rgb, aug_seg = apply_random_augmentation(rgb_img.copy(), seg_img.copy())

                rgb_aug_out = os.path.join(
                    rgb_out_dir, f"{name}_aug{a}.png"
                )
                save_image(aug_rgb, rgb_aug_out)

                if aug_seg is not None:
                    seg_aug_out = os.path.join(
                        lbl_out_dir, f"{name}_aug{a}.png"
                    )
                    save_image(aug_seg, seg_aug_out)

                # YOLO label for augmented data:
                if os.path.exists(yolo_src):
                    copy_file(yolo_src, os.path.join(lbl_out_dir, f"{name}_aug{a}.txt"))

# =====================================================================================
# Main
# =====================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_root", help="Root folder containing images/ and labels/")
    parser.add_argument("out_root", help="Output YOLO dataset root")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--valid", type=float, default=0.15)
    parser.add_argument("--test",  type=float, default=0.15)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    augment_enabled = not args.no_augment

    print("===============================================")
    print("      YOLO Dataset Split & Augmentation        ")
    print("===============================================")
    print("Source root:      ", args.src_root)
    print("Output root:      ", args.out_root)
    print("Train/Val/Test:   ", args.train, args.valid, args.test)
    print("Multiplier:       ", args.multiplier)
    print("Augmentation:     ", "ON" if augment_enabled else "OFF")
    print("===============================================")

    # Collect files
    img_dir = os.path.join(args.src_root, "images")
    all_images = sorted([os.path.join("images", f) for f in os.listdir(img_dir)
                         if f.endswith(".png")])

    if len(all_images) == 0:
        print("❌ No images found in images/")
        return

    # Split
    train_idx, val_idx, test_idx = split_indices(
        len(all_images), args.train, args.valid, args.test
    )

    # Process subsets
    process_set(train_idx, all_images, args.src_root, args.out_root,
                "train", args.multiplier, augment_enabled)

    process_set(val_idx, all_images, args.src_root, args.out_root,
                "valid", 1.0, False)

    process_set(test_idx, all_images, args.src_root, args.out_root,
                "test", 1.0, False)

    print("===============================================")
    print("✅ YOLO dataset creation complete!")
    print("Output:", args.out_root)
    print("===============================================")


if __name__ == "__main__":
    main()
