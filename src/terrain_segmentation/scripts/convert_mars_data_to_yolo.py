#!/usr/bin/env python3
"""
Convert AI4Mars merged dataset to YOLO semantic segmentation format.
Handles inconsistent directory structure by finding labels first, then matching images.
Generates both PNG masks and YOLO text files with polygon coordinates.
Usage: python convert_ai4mars_to_yolo.py /path/to/ai4mars-dataset-merged-0.6
https://zenodo.org/records/15995036
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


def mask_to_yolo_segments(mask, class_id):
    """Convert a binary mask for a specific class to YOLO polygon format."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    segments = []
    height, width = mask.shape
    
    for contour in contours:
        if len(contour) < 3:
            continue
            
        contour = contour.reshape(-1, 2)
        
        if len(contour) < 3:
            continue
        
        normalized = contour.astype(float)
        normalized[:, 0] /= width
        normalized[:, 1] /= height
        
        segment = [class_id] + normalized.flatten().tolist()
        segments.append(segment)
    
    return segments


def convert_mask_to_yolo_txt(class_label, output_txt_path):
    """Convert a class ID mask to YOLO text format."""
    unique_classes = np.unique(class_label)
    unique_classes = unique_classes[(unique_classes < 255) & (unique_classes >= 0)]
    
    all_segments = []
    
    for class_id in unique_classes:
        class_mask = (class_label == class_id).astype(np.uint8)
        segments = mask_to_yolo_segments(class_mask, int(class_id))
        all_segments.extend(segments)
    
    with open(output_txt_path, 'w') as f:
        for segment in all_segments:
            line = ' '.join(map(str, segment))
            f.write(line + '\n')
    
    return len(all_segments) > 0


def find_matching_image(label_path, root_dir):
    """
    Find the matching image for a label file.
    Searches in multiple possible locations based on the label path structure.
    """
    image_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    base_name = label_path.stem
    
    # Strip merged suffixes (_merged13, _merged14, etc.)
    import re
    base_name = re.sub(r'_merged\d+$', '', base_name)
    
    # Get the relative path from root to understand structure
    try:
        rel_path = label_path.relative_to(root_dir)
        parts = rel_path.parts  # e.g., ('mer', 'labels', 'train', 'filename.png')
    except ValueError:
        return None
    
    # Determine mission and camera type from path
    mission = parts[0] if len(parts) > 0 else None
    
    # Build list of potential image directories to search
    search_dirs = []
    
    if mission:
        mission_path = root_dir / mission
        
        # Common patterns:
        # 1. mission/images/edr/
        # 2. mission/camera/images/edr/ (e.g., msl/ncam/images/edr/)
        # 3. mission/camera/images/ (e.g., msl/mcam/images/)
        # 4. mission/images/camera/ (e.g., m2020/images/ncam/)
        
        # Pattern 1: mission/images/
        images_base = mission_path / 'images'
        if images_base.exists():
            search_dirs.append(images_base)
            # Also check edr subdirectory
            if (images_base / 'edr').exists():
                search_dirs.append(images_base / 'edr')
            # Check for camera subdirectories
            for cam in ['ncam', 'mcam', 'HAFIQ']:
                if (images_base / cam).exists():
                    search_dirs.append(images_base / cam)
                    if (images_base / cam / 'edr').exists():
                        search_dirs.append(images_base / cam / 'edr')
        
        # Pattern 2 & 3: mission/camera/images/
        for cam in ['ncam', 'mcam', 'HAFIQ']:
            cam_images = mission_path / cam / 'images'
            if cam_images.exists():
                search_dirs.append(cam_images)
                if (cam_images / 'edr').exists():
                    search_dirs.append(cam_images / 'edr')
    
    # Search for the image in all potential directories
    for search_dir in search_dirs:
        for ext in image_extensions:
            img_path = search_dir / f"{base_name}{ext}"
            if img_path.exists():
                return img_path
    
    return None


def find_all_labels(root_dir):
    """Find all label files recursively."""
    labels = []
    root_dir = Path(root_dir)
    
    # Search for labels in all missions
    for mission in ['m2020', 'mer', 'msl']:
        mission_path = root_dir / mission
        if not mission_path.exists():
            continue
        
        # Find all PNG files in labels directories
        for label_path in mission_path.rglob('*.png'):
            # Check if it's in a labels directory
            if 'labels' in str(label_path):
                labels.append(label_path)
    
    return labels


def process_dataset(root_dir, output_dir, max_files=None):
    """Process the entire dataset."""
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    
    print(f"Scanning dataset in: {root_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Generating both PNG masks AND YOLO text files...\n")
    
    # Find all labels
    print("Finding all label files...")
    all_labels = find_all_labels(root_dir)
    print(f"  Found {len(all_labels)} label files\n")
    
    if len(all_labels) == 0:
        print("Error: No label files found!")
        return
    
    # Limit if requested
    if max_files is not None and max_files > 0:
        all_labels = all_labels[:max_files]
        print(f"Limited to {len(all_labels)} labels\n")
    
    # Create output directories
    output_labels_dir = output_dir / 'labels'
    output_images_dir = output_dir / 'images'
    
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all labels and find matching images
    stats = {class_id: 0 for class_id in RGB_TO_CLASS.values()}
    successful = 0
    failed = 0
    no_image = 0
    
    print(f"Processing {len(all_labels)} labels...")
    
    for label_path in tqdm(all_labels, desc="Converting"):
        try:
            # Find matching image
            img_path = find_matching_image(label_path, root_dir)
            
            if img_path is None:
                no_image += 1
                continue
            
            # Determine mission and split from path
            rel_path = label_path.relative_to(root_dir)
            parts = rel_path.parts
            mission = parts[0] if len(parts) > 0 else "unknown"
            
            # Check if it's train or test split
            split = "train" if "train" in str(label_path) else "test"
            
            # Create unique filename
            base_name = label_path.stem
            output_name = f"{mission}_{split}_{base_name}"
            
            # Load and convert label
            rgb_label = np.array(Image.open(label_path))
            class_label = rgb_to_class_id(rgb_label)
            
            # Update stats
            unique, counts = np.unique(class_label, return_counts=True)
            for class_id, count_pixels in zip(unique, counts):
                if class_id in stats:
                    stats[class_id] += count_pixels
            
            # Save converted label PNG
            output_png_path = output_labels_dir / f"{output_name}.png"
            Image.fromarray(class_label, mode='L').save(output_png_path)
            
            # Generate and save YOLO text file
            output_txt_path = output_labels_dir / f"{output_name}.txt"
            convert_mask_to_yolo_txt(class_label, output_txt_path)
            
            # Copy image
            import shutil
            output_img_path = output_images_dir / f"{output_name}{img_path.suffix}"
            shutil.copy(img_path, output_img_path)
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {label_path.name}: {e}")
            failed += 1
            continue
    
    # Print statistics
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"\nTotal labels found: {len(all_labels)}")
    print(f"Successful conversions: {successful}")
    print(f"No matching image: {no_image}")
    print(f"Failed: {failed}")
    
    print("\nClass distribution:")
    total_pixels = sum(stats.values())
    for class_id, count in sorted(stats.items()):
        class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
        percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  {class_name} (ID {class_id}): {count:,} pixels ({percentage:.2f}%)")
    
    # Create data.yaml
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"# AI4Mars Terrain Semantic Segmentation Dataset (Merged)\n")
        f.write(f"# All missions (M2020, MER, MSL) combined into single dataset\n\n")
        f.write(f"path: {output_dir.absolute()}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n\n")
        f.write(f"names:\n")
        for class_id in range(4):
            f.write(f"  {class_id}: {CLASS_NAMES[class_id]}\n")
        f.write(f"\n# Files are prefixed with: mission_split_filename\n")
        f.write(f"# You can split this dataset as needed for your training\n")
    
    print(f"\nOutput saved to: {output_dir}")
    print(f"  Labels (PNG): {len(list(output_labels_dir.glob('*.png')))} files")
    print(f"  Labels (TXT): {len(list(output_labels_dir.glob('*.txt')))} files")
    print(f"  Images: {len(list(output_images_dir.glob('*')))} files")
    print(f"  Config: {yaml_path}")
    print("\nBoth PNG masks and YOLO text files have been generated!")
    print("You can now use this dataset for YOLO training!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_ai4mars_to_yolo.py /path/to/ai4mars-dataset-merged-0.6 [--output /path/to/output] [--limit N]")
        print("\nOptions:")
        print("  --output PATH    Output directory (default: ai4mars-dataset-merged-0.6/AI4MARS_YOLO)")
        print("  --limit N        Process only N files")
        print("\nExamples:")
        print("  python convert_ai4mars_to_yolo.py /path/to/ai4mars-dataset-merged-0.6")
        print("  python convert_ai4mars_to_yolo.py /path/to/ai4mars-dataset-merged-0.6 --limit 100")
        print("  python convert_ai4mars_to_yolo.py /path/to/ai4mars-dataset-merged-0.6 --output /path/to/output")
        sys.exit(1)
    
    dataset_path = Path(sys.argv[1])
    output_path = None
    max_files = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_path = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--limit' and i + 1 < len(sys.argv):
            try:
                max_files = int(sys.argv[i + 1])
                print(f"Limit set to {max_files} files\n")
            except ValueError:
                print(f"Error: --limit must be followed by a number")
                sys.exit(1)
            i += 2
        else:
            i += 1
    
    # Set default output path if not specified
    if output_path is None:
        output_path = dataset_path / 'AI4MARS_YOLO'
    
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    process_dataset(dataset_path, output_path, max_files)


if __name__ == '__main__':
    main()