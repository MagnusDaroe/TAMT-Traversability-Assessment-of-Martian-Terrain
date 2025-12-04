"""
Verify that image and mask filenames match correctly for YOLO
"""

from pathlib import Path
import sys


def verify_image_mask_pairing(images_dir, masks_dir):
    """
    Check if image and mask files have matching filenames.
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
    """
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    
    print("=" * 60)
    print("IMAGE-MASK PAIRING VERIFICATION")
    print("=" * 60)
    
    # Get all image files
    image_files = {}
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        for f in images_path.glob(ext):
            # Store by stem (filename without extension)
            image_files[f.stem] = f
    
    # Get all mask files
    mask_files = {}
    for f in masks_path.glob('*.png'):
        mask_files[f.stem] = f
    
    print(f"\nImages directory: {images_path}")
    print(f"  Total images: {len(image_files)}")
    
    print(f"\nMasks directory: {masks_path}")
    print(f"  Total masks: {len(mask_files)}")
    
    # Find matches and mismatches
    image_stems = set(image_files.keys())
    mask_stems = set(mask_files.keys())
    
    matched = image_stems & mask_stems
    images_without_masks = image_stems - mask_stems
    masks_without_images = mask_stems - image_stems
    
    print(f"\n" + "=" * 60)
    print("PAIRING RESULTS")
    print("=" * 60)
    print(f"✓ Matched pairs: {len(matched)}")
    print(f"⚠ Images without masks: {len(images_without_masks)}")
    print(f"⚠ Masks without images: {len(masks_without_images)}")
    
    # Show sample matches
    if matched:
        print(f"\nSample matched pairs (first 5):")
        for stem in list(matched)[:5]:
            img_file = image_files[stem]
            mask_file = mask_files[stem]
            print(f"  ✓ {img_file.name} <-> {mask_file.name}")
    
    # Show mismatches
    if images_without_masks:
        print(f"\nSample images without masks (first 10):")
        for stem in list(images_without_masks)[:10]:
            print(f"  ⚠ {image_files[stem].name}")
    
    if masks_without_images:
        print(f"\nSample masks without images (first 10):")
        for stem in list(masks_without_images)[:10]:
            print(f"  ⚠ {mask_files[stem].name}")
    
    # Check if there's a pattern in mismatches
    if images_without_masks or masks_without_images:
        print(f"\n" + "=" * 60)
        print("MISMATCH ANALYSIS")
        print("=" * 60)
        
        # Sample filenames
        if images_without_masks:
            sample_img = image_files[list(images_without_masks)[0]].name
            print(f"\nSample image name: {sample_img}")
        
        if masks_without_images:
            sample_mask = mask_files[list(masks_without_images)[0]].name
            print(f"Sample mask name: {sample_mask}")
        
        print("\n⚠️  ISSUE: Filenames don't match!")
        print("For YOLO segmentation, image and mask must have the same base name:")
        print("  images/train/image123.jpg")
        print("  labels/train/image123.png  ← Same base name!")
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if len(matched) == len(image_files) == len(mask_files):
        print("✅ Perfect! All images have matching masks.")
        print("   Your dataset is properly paired for YOLO training.")
    elif len(matched) > 0:
        match_rate = (len(matched) / len(image_files)) * 100
        print(f"⚠️  Only {match_rate:.1f}% of images have matching masks.")
        print("   You need to fix the filename pairing.")
    else:
        print("❌ No matching pairs found!")
        print("   Image and mask filenames don't align at all.")
        print("   You need to rename files so they match.")


def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_pairing.py <images_dir> <masks_dir>")
        print("Example: python verify_pairing.py ./dataset/images/train ./dataset/labels/train")
        return
    
    images_dir = sys.argv[1]
    masks_dir = sys.argv[2]
    
    if not Path(images_dir).exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not Path(masks_dir).exists():
        print(f"Error: Masks directory not found: {masks_dir}")
        return
    
    verify_image_mask_pairing(images_dir, masks_dir)


if __name__ == '__main__':
    main()