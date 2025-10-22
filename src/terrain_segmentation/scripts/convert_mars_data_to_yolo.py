#!/usr/bin/env python3
"""
Convert Mars terrain dataset from RGB labels to YOLO semantic segmentation format.
Merges train and test into a single dataset without splits.
Generates both PNG masks and YOLO text files with polygon coordinates.
Usage: python convert_mars_to_yolo.py /path/to/dataset
"""

import sys
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2


# RGB to class mapping
RGB_TO_CLASS = {
    (0, 0, 0): 0,       # soil
    (1, 1, 1): 1,       # bedrock
    (2, 2, 2): 2,       # sand
    (3, 3, 3): 3,       # big rock
    (255, 255, 255): 255  # NULL
}

CLASS_NAMES = {0: 'soil', 1: 'bedrock', 2: 'sand', 3: 'big_rock', 255: 'null'}


def rgb_to_class_id(rgb_label):
    """Convert RGB label to class ID format."""
    # Handle grayscale images - convert to RGB if needed
    if len(rgb_label.shape) == 2:
        # Grayscale image, expand to RGB
        rgb_label = np.stack([rgb_label, rgb_label, rgb_label], axis=-1)
    
    height, width = rgb_label.shape[:2]
    class_label = np.full((height, width), 255, dtype=np.uint8)
    
    for rgb_value, class_id in RGB_TO_CLASS.items():
        mask = np.all(rgb_label == rgb_value, axis=-1)
        class_label[mask] = class_id
    
    return class_label


def apply_masks(class_label, rover_mask_path, range_mask_path):
    """Apply rover and range masks."""
    masked_label = class_label.copy()
    
    if rover_mask_path.exists():
        rover_mask = np.array(Image.open(rover_mask_path))
        if len(rover_mask.shape) > 2:
            rover_mask = rover_mask[:, :, 0]
        masked_label[rover_mask == 1] = 255
    
    if range_mask_path.exists():
        range_mask = np.array(Image.open(range_mask_path))
        if len(range_mask.shape) > 2:
            range_mask = range_mask[:, :, 0]
        masked_label[range_mask == 1] = 255
    
    return masked_label


def mask_to_yolo_segments(mask, class_id):
    """
    Convert a binary mask for a specific class to YOLO polygon format.
    
    Args:
        mask: Binary mask (H, W) where True/1 indicates the class
        class_id: Integer class ID
        
    Returns:
        List of normalized polygon coordinates [class_id, x1, y1, x2, y2, ...]
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    segments = []
    height, width = mask.shape
    
    for contour in contours:
        # Skip very small contours (noise)
        if len(contour) < 3:
            continue
            
        # Flatten and normalize coordinates
        contour = contour.reshape(-1, 2)
        
        # Skip if too few points
        if len(contour) < 3:
            continue
        
        # Normalize to 0-1 range
        normalized = contour.astype(float)
        normalized[:, 0] /= width   # x coordinates
        normalized[:, 1] /= height  # y coordinates
        
        # Create YOLO format: class_id x1 y1 x2 y2 ...
        segment = [class_id] + normalized.flatten().tolist()
        segments.append(segment)
    
    return segments


def convert_mask_to_yolo_txt(class_label, output_txt_path):
    """
    Convert a class ID mask to YOLO text format.
    
    Args:
        class_label: Numpy array with class IDs
        output_txt_path: Path to output .txt file
    """
    # Get unique class values (excluding background/255)
    unique_classes = np.unique(class_label)
    unique_classes = unique_classes[(unique_classes < 255) & (unique_classes >= 0)]
    
    all_segments = []
    
    # Process each class
    for class_id in unique_classes:
        # Create binary mask for this class
        class_mask = (class_label == class_id).astype(np.uint8)
        
        # Convert to YOLO segments
        segments = mask_to_yolo_segments(class_mask, int(class_id))
        all_segments.extend(segments)
    
    # Write to file
    with open(output_txt_path, 'w') as f:
        for segment in all_segments:
            # Format: class_id x1 y1 x2 y2 x3 y3 ...
            line = ' '.join(map(str, segment))
            f.write(line + '\n')
    
    return len(all_segments) > 0


def process_split(split_name, labels_dir, images_base, output_labels_dir, output_images_dir, stats, max_files=None):
    """Process a single split (train or test)."""
    # Images are in images/edr/, images/mxy/, images/rng-30m/
    # NOT in images/train/edr/ - they're shared across train/test
    images_dir = images_base / 'edr' if (images_base / 'edr').exists() else images_base
    rover_masks_dir = images_base / 'mxy'
    range_masks_dir = images_base / 'rng-30m'
    
    if not labels_dir.exists():
        print(f"  Warning: {labels_dir} not found, skipping...")
        return 0, 0, 0
    
    # Get all label files
    label_files = sorted(labels_dir.glob('*.png'))
    
    if len(label_files) == 0:
        print(f"  Warning: No label files found in {labels_dir}")
        return 0, 0, 0
    
    # Limit number of files if specified
    if max_files is not None and max_files > 0:
        label_files = label_files[:max_files]
        print(f"  Processing {len(label_files)} files from {split_name} (limited to {max_files})...")
    else:
        print(f"  Processing {len(label_files)} files from {split_name}...")
    
    count = 0
    skipped = 0
    
    for label_path in tqdm(label_files, desc=f"  Converting {split_name}"):
        base_name = label_path.stem
        
        # Check if matching image exists FIRST
        img_path = None
        if images_dir.exists():
            for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG']:
                potential_img = images_dir / f"{base_name}{ext}"
                if potential_img.exists():
                    img_path = potential_img
                    break
        
        # Skip this label if no matching image found
        if img_path is None:
            skipped += 1
            continue
        
        # Load and convert label
        rgb_label = np.array(Image.open(label_path))
        class_label = rgb_to_class_id(rgb_label)
        
        # Apply masks
        rover_mask_path = rover_masks_dir / f"{base_name}.png"
        range_mask_path = range_masks_dir / f"{base_name}.png"
        class_label = apply_masks(class_label, rover_mask_path, range_mask_path)
        
        # Update stats
        unique, counts = np.unique(class_label, return_counts=True)
        for class_id, count_pixels in zip(unique, counts):
            if class_id in stats:
                stats[class_id] += count_pixels
        
        # Save converted label PNG with prefix
        output_png_path = output_labels_dir / f"{split_name}_{base_name}.png"
        Image.fromarray(class_label, mode='L').save(output_png_path)
        
        # Generate and save YOLO text file
        output_txt_path = output_labels_dir / f"{split_name}_{base_name}.txt"
        convert_mask_to_yolo_txt(class_label, output_txt_path)
        
        # Copy image
        import shutil
        shutil.copy(img_path, output_images_dir / f"{split_name}_{img_path.name}")
        count += 1
    
    return count, len(label_files), skipped


def convert_dataset(root_dir, max_files=None):
    """Convert the entire dataset, merging train and test."""
    root_dir = Path(root_dir)
    
    print(f"Converting dataset in: {root_dir}")
    if max_files:
        print(f"Limiting to {max_files} files per split")
    print(f"Merging train and test splits into single dataset...")
    print(f"Generating both PNG masks AND YOLO text files...\n")
    
    # Check for train/test structure
    labels_base = root_dir / 'labels'
    images_base = root_dir / 'images'
    
    if not labels_base.exists():
        print(f"Error: Labels directory not found: {labels_base}")
        print(f"\nExpected structure:")
        print(f"  {root_dir}/")
        print(f"    labels/")
        print(f"      train/")
        print(f"      test/")
        print(f"    images/")
        print(f"      edr/")
        print(f"      mxy/")
        print(f"      rng-30m/")
        return
    
    # Create output directory (no cleanup - add to existing)
    output_dir = root_dir / 'AI4MARS_YOLO'
    output_labels_dir = output_dir / 'labels'
    output_images_dir = output_dir / 'images'
    
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output will be saved to: {output_dir}\n")
    
    stats = {class_id: 0 for class_id in RGB_TO_CLASS.values()}
    total_files = 0
    total_labels = 0
    total_skipped = 0
    
    # Process train split
    train_labels = labels_base / 'train'
    if train_labels.exists():
        print("Processing TRAIN split:")
        count, labels_count, skipped = process_split('train', train_labels, images_base, 
                            output_labels_dir, output_images_dir, stats, max_files)
        total_files += count
        total_labels += labels_count
        total_skipped += skipped
        if skipped > 0:
            print(f"  ⚠ Skipped {skipped} labels without matching images")
        print(f"  ✓ Processed {count} matched pairs\n")
    
    # Process test split
    test_labels = labels_base / 'test'
    if test_labels.exists():
        print("Processing TEST split:")
        count, labels_count, skipped = process_split('test', test_labels, images_base, 
                            output_labels_dir, output_images_dir, stats, max_files)
        total_files += count
        total_labels += labels_count
        total_skipped += skipped
        if skipped > 0:
            print(f"  ⚠ Skipped {skipped} labels without matching images")
        print(f"  ✓ Processed {count} matched pairs\n")
    
    # Print statistics
    print("=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"\nTotal labels found: {total_labels}")
    print(f"Total pairs processed: {total_files}")
    print(f"Total skipped (no image): {total_skipped}")
    
    print("\nClass distribution:")
    total_pixels = sum(stats.values())
    for class_id, count in sorted(stats.items()):
        class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
        percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  {class_name} (ID {class_id}): {count:,} pixels ({percentage:.2f}%)")
    
    # Create dataset.yaml
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"# Mars Terrain Semantic Segmentation Dataset (Merged)\n")
        f.write(f"# All train and test data combined - split yourself as needed\n\n")
        f.write(f"path: {output_dir.absolute()}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n\n")
        f.write(f"names:\n")
        for class_id in range(4):
            f.write(f"  {class_id}: {CLASS_NAMES[class_id]}\n")
        f.write(f"\n# Note: All files are prefixed with 'train_' or 'test_' to indicate origin\n")
        f.write(f"# You can split this dataset as needed for your training\n")
    
    print(f"\nOutput saved to: {output_dir}")
    print(f"  Labels (PNG): {output_labels_dir} ({len(list(output_labels_dir.glob('*.png')))} files)")
    print(f"  Labels (TXT): {output_labels_dir} ({len(list(output_labels_dir.glob('*.txt')))} files)")
    print(f"  Images: {output_images_dir} ({len(list(output_images_dir.glob('*')))} files)")
    print(f"  Config: {yaml_path}")
    print("\nNote: Files are prefixed with 'train_' or 'test_' to indicate their origin.")
    print("Both PNG masks and YOLO text files have been generated!")
    print("You can now use this dataset for YOLO training!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_mars_to_yolo.py /path/to/dataset [--limit N]")
        print("\nOptions:")
        print("  --limit N    Process only N files per split (train/test)")
        print("\nExamples:")
        print("  python convert_mars_to_yolo.py /path/to/dataset")
        print("  python convert_mars_to_yolo.py /path/to/dataset --limit 100")
        print("\nExpected directory structure:")
        print("  dataset/")
        print("    labels/")
        print("      train/          # Train RGB labels")
        print("      test/           # Test RGB labels")
        print("    images/")
        print("      edr/            # All images (shared between train/test)")
        print("      mxy/            # Rover masks (optional)")
        print("      rng-30m/        # Range masks (optional)")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    max_files = None
    
    # Parse --limit argument
    if len(sys.argv) >= 4 and sys.argv[2] == '--limit':
        try:
            max_files = int(sys.argv[3])
            print(f"Limit set to {max_files} files per split\n")
        except ValueError:
            print(f"Error: --limit must be followed by a number")
            sys.exit(1)
    
    convert_dataset(dataset_path, max_files)