#!/usr/bin/env python3
"""
Convert AI4Mars merged dataset to YOLO semantic segmentation format.
Creates two dataset configurations:
  1. NAV (Navigation/Traversability): 4 classes for terrain traversability
  2. GEO (Geology): 27+ classes for geological classification (M2020 only)

Handles inconsistent directory structure by finding labels first, then matching images.
Generates both PNG masks and YOLO text files with polygon coordinates.

Usage: python convert_ai4mars_to_yolo_dual.py /path/to/ai4mars-dataset-merged-0.6 --config [nav|geo|both]
https://zenodo.org/records/15995036
"""

import sys
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2


# =============================================================================
# NAV (Navigation/Traversability) Configuration
# =============================================================================
NAV_RGB_TO_CLASS = {
    (0, 0, 0): 0,       # soil
    (1, 1, 1): 1,       # bedrock
    (2, 2, 2): 2,       # sand
    (3, 3, 3): 3,       # big rock
    (255, 255, 255): 255  # NULL
}

NAV_CLASS_NAMES = {
    0: 'soil', 
    1: 'bedrock', 
    2: 'sand', 
    3: 'big_rock', 
    255: 'null'
}

# =============================================================================
# GEO (Geology) Configuration
# =============================================================================
GEO_RGB_TO_CLASS = {
    # Bedrock types (0-6)
    (0, 0, 0): 0,       # bedrock_massive
    (1, 1, 1): 1,       # bedrock_layered_angled
    (2, 2, 2): 2,       # bedrock_layered_flat
    (3, 3, 3): 3,       # bedrock_layered_unsure
    (4, 4, 4): 4,       # bedrock_conglomerate
    (5, 5, 5): 5,       # bedrock_holey
    (6, 6, 6): 6,       # bedrock_unsure
    
    # Float rocks (10-17)
    (10, 10, 10): 10,   # float_rock_massive
    (11, 11, 11): 11,   # float_rock_layered_angled
    (12, 12, 12): 12,   # float_rock_layered_flat
    (13, 13, 13): 13,   # float_rock_layered_unsure
    (14, 14, 14): 14,   # float_rock_conglomerate
    (15, 15, 15): 15,   # float_rock_holey
    (16, 16, 16): 16,   # float_rock_mixed
    (17, 17, 17): 17,   # float_rock_unsure
    
    # Sand types (20-22)
    (20, 20, 20): 20,   # sand_dune
    (21, 21, 21): 21,   # sand_ripples
    (22, 22, 22): 22,   # sand_sand
    
    # Other geological features (30-50)
    (30, 30, 30): 30,   # pebbles
    (40, 40, 40): 40,   # vein
    (50, 50, 50): 50,   # hill_peak
    
    (255, 255, 255): 255  # NULL
}

GEO_CLASS_NAMES = {
    0: 'bedrock_massive',
    1: 'bedrock_layered_angled',
    2: 'bedrock_layered_flat',
    3: 'bedrock_layered_unsure',
    4: 'bedrock_conglomerate',
    5: 'bedrock_holey',
    6: 'bedrock_unsure',
    10: 'float_rock_massive',
    11: 'float_rock_layered_angled',
    12: 'float_rock_layered_flat',
    13: 'float_rock_layered_unsure',
    14: 'float_rock_conglomerate',
    15: 'float_rock_holey',
    16: 'float_rock_mixed',
    17: 'float_rock_unsure',
    20: 'sand_dune',
    21: 'sand_ripples',
    22: 'sand_sand',
    30: 'pebbles',
    40: 'vein',
    50: 'hill_peak',
    255: 'null'
}


def rgb_to_class_id(rgb_label, label_type='nav'):
    """Convert RGB label to class ID format."""
    # Handle grayscale images - convert to RGB if needed
    if len(rgb_label.shape) == 2:
        # Grayscale image, expand to RGB
        rgb_label = np.stack([rgb_label, rgb_label, rgb_label], axis=-1)
    
    height, width = rgb_label.shape[:2]
    class_label = np.full((height, width), 255, dtype=np.uint8)
    
    # Select the appropriate mapping
    rgb_to_class = NAV_RGB_TO_CLASS if label_type == 'nav' else GEO_RGB_TO_CLASS
    
    for rgb_value, class_id in rgb_to_class.items():
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


def detect_label_type(label_path):
    """
    Detect whether a label is NAV or GEO type based on directory structure.
    Returns: 'nav' or 'geo'
    """
    path_str = str(label_path)
    
    # M2020_GEO labels are in M2020_GEO directory
    if 'M2020_GEO' in path_str or 'm2020_geo' in path_str.lower():
        return 'geo'
    
    # NAV labels are in NAV directory
    if '/NAV/' in path_str or '/nav/' in path_str.lower():
        return 'nav'
    
    # Default: all MER and MSL labels are NAV
    # M2020 labels outside M2020_GEO are also NAV
    return 'nav'


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
        parts = rel_path.parts  # e.g., ('m2020', 'labels', 'NAV', 'filename.png')
    except ValueError:
        return None
    
    # Determine mission from path
    mission = parts[0].lower() if len(parts) > 0 else None
    
    # Build list of potential image directories to search
    search_dirs = []
    
    if mission:
        mission_path = root_dir / parts[0]  # Preserve original case
        
        # M2020 specific patterns
        if mission == 'm2020':
            images_base = mission_path / 'images'
            if images_base.exists():
                search_dirs.append(images_base)
                # Check for camera subdirectories
                for cam in ['HAFIQ', 'mcam', 'ncam']:
                    if (images_base / cam).exists():
                        search_dirs.append(images_base / cam)
        
        # MER specific patterns
        elif mission == 'mer':
            images_base = mission_path / 'images'
            if images_base.exists():
                search_dirs.append(images_base)
                for subdir in ['eff', 'test']:
                    if (images_base / subdir).exists():
                        search_dirs.append(images_base / subdir)
        
        # MSL specific patterns
        elif mission == 'msl':
            # Pattern: msl/camera/images/
            for cam in ['ncam', 'mcam']:
                cam_images = mission_path / cam / 'images'
                if cam_images.exists():
                    search_dirs.append(cam_images)
                    # Check subdirectories
                    for subdir in ['edr', 'mxy', 'rng-30m']:
                        if (cam_images / subdir).exists():
                            search_dirs.append(cam_images / subdir)
    
    # Search for the image in all potential directories
    for search_dir in search_dirs:
        for ext in image_extensions:
            img_path = search_dir / f"{base_name}{ext}"
            if img_path.exists():
                return img_path
    
    return None


def find_all_labels(root_dir, config='both'):
    """
    Find all label files recursively.
    config: 'nav', 'geo', or 'both'
    Returns dict: {'nav': [...], 'geo': [...]}
    """
    labels = {'nav': [], 'geo': []}
    root_dir = Path(root_dir)
    
    # Search for labels in all missions
    for mission in ['m2020', 'M2020', 'mer', 'msl']:
        mission_path = root_dir / mission
        if not mission_path.exists():
            continue
        
        # Find all PNG files in labels directories
        for label_path in mission_path.rglob('*.png'):
            # Check if it's in a labels directory
            path_str = str(label_path)
            if 'labels' not in path_str.lower():
                continue
            
            # Skip certain directories
            if 'raw_unmerged' in path_str:
                continue
            
            # For test sets with multiple agreement levels, only use min3-100agree
            if 'masked-gold' in path_str:
                if 'min3-100agree' not in path_str:
                    continue
            
            # Detect label type
            label_type = detect_label_type(label_path)
            
            # Add to appropriate list based on config
            if config == 'both' or config == label_type:
                labels[label_type].append(label_path)
    
    return labels


def process_dataset(root_dir, output_dir, config='both', max_files=None):
    """
    Process the entire dataset.
    config: 'nav', 'geo', or 'both'
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    
    print(f"Scanning dataset in: {root_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Configuration: {config.upper()}")
    print(f"Generating both PNG masks AND YOLO text files...\n")
    
    # Find all labels
    print("Finding all label files...")
    all_labels = find_all_labels(root_dir, config)
    
    for label_type in ['nav', 'geo']:
        if all_labels[label_type]:
            print(f"  Found {len(all_labels[label_type])} {label_type.upper()} label files")
    print()
    
    if sum(len(labels) for labels in all_labels.values()) == 0:
        print("Error: No label files found!")
        return
    
    # Process each configuration
    for label_type in ['nav', 'geo']:
        if not all_labels[label_type]:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {label_type.upper()} dataset")
        print(f"{'='*60}\n")
        
        # Limit if requested
        labels_to_process = all_labels[label_type]
        if max_files is not None and max_files > 0:
            labels_to_process = labels_to_process[:max_files]
            print(f"Limited to {len(labels_to_process)} labels\n")
        
        # Create output directories for this configuration
        config_output_dir = output_dir / f'ai4mars_{label_type}'
        output_labels_dir = config_output_dir / 'labels'
        output_images_dir = config_output_dir / 'images'
        
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Get class names for this configuration
        class_names = NAV_CLASS_NAMES if label_type == 'nav' else GEO_CLASS_NAMES
        rgb_to_class = NAV_RGB_TO_CLASS if label_type == 'nav' else GEO_RGB_TO_CLASS
        
        # Process all labels
        stats = {class_id: 0 for class_id in rgb_to_class.values()}
        successful = 0
        failed = 0
        no_image = 0
        
        print(f"Processing {len(labels_to_process)} labels...")
        
        for label_path in tqdm(labels_to_process, desc=f"Converting {label_type.upper()}"):
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
                split = "train" if "train" in str(label_path).lower() else "test"
                
                # Create unique filename
                base_name = label_path.stem
                output_name = f"{mission}_{split}_{base_name}"
                
                # Load and convert label
                rgb_label = np.array(Image.open(label_path))
                class_label = rgb_to_class_id(rgb_label, label_type)
                
                # Get label dimensions (NOT image dimensions!)
                label_height, label_width = class_label.shape
                
                # Update stats
                unique, counts = np.unique(class_label, return_counts=True)
                for class_id, count_pixels in zip(unique, counts):
                    if class_id in stats:
                        stats[class_id] += count_pixels
                
                # Save converted label PNG
                output_png_path = output_labels_dir / f"{output_name}.png"
                Image.fromarray(class_label, mode='L').save(output_png_path)
                
                # Generate and save YOLO text file (using label dimensions)
                output_txt_path = output_labels_dir / f"{output_name}.txt"
                convert_mask_to_yolo_txt(class_label, output_txt_path)
                
                # Load and resize image to match label dimensions
                image = Image.open(img_path)
                image_resized = image.resize((label_width, label_height), Image.LANCZOS)
                
                # Save resized image
                output_img_path = output_images_dir / f"{output_name}{img_path.suffix}"
                image_resized.save(output_img_path)
                
                successful += 1
                
            except Exception as e:
                print(f"\nError processing {label_path.name}: {e}")
                failed += 1
                continue
        
        # Print statistics for this configuration
        print("\n" + "=" * 60)
        print(f"{label_type.upper()} CONVERSION COMPLETE!")
        print("=" * 60)
        print(f"\nTotal labels found: {len(labels_to_process)}")
        print(f"Successful conversions: {successful}")
        print(f"No matching image: {no_image}")
        print(f"Failed: {failed}")
        
        print("\nClass distribution:")
        total_pixels = sum(stats.values())
        for class_id, count in sorted(stats.items()):
            class_name = class_names.get(class_id, f"unknown_{class_id}")
            percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
            print(f"  {class_name} (ID {class_id}): {count:,} pixels ({percentage:.2f}%)")
        
        # Create data.yaml
        yaml_path = config_output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            if label_type == 'nav':
                f.write(f"# AI4Mars Navigation/Traversability Dataset\n")
                f.write(f"# Terrain classification for rover navigation\n")
                f.write(f"# Missions: M2020, MER, MSL\n\n")
            else:
                f.write(f"# AI4Mars Geology Dataset\n")
                f.write(f"# Detailed geological terrain classification\n")
                f.write(f"# Mission: M2020 only\n\n")
            
            f.write(f"path: {config_output_dir.absolute()}\n")
            f.write(f"train: images\n")
            f.write(f"val: images\n\n")
            f.write(f"names:\n")
            
            # Write class names (only non-null classes)
            valid_classes = sorted([cid for cid in class_names.keys() if cid < 255])
            for class_id in valid_classes:
                f.write(f"  {class_id}: {class_names[class_id]}\n")
            
            f.write(f"\n# Files are prefixed with: mission_split_filename\n")
            f.write(f"# You can split this dataset as needed for your training\n")
        
        print(f"\nOutput saved to: {config_output_dir}")
        print(f"  Labels (PNG): {len(list(output_labels_dir.glob('*.png')))} files")
        print(f"  Labels (TXT): {len(list(output_labels_dir.glob('*.txt')))} files")
        print(f"  Images: {len(list(output_images_dir.glob('*')))} files")
        print(f"  Config: {yaml_path}")
    
    print("\n" + "=" * 60)
    print("ALL CONVERSIONS COMPLETE!")
    print("=" * 60)
    print("\nDatasets created:")
    if all_labels['nav']:
        print(f"  - ai4mars_nav: Navigation/Traversability (4 classes)")
    if all_labels['geo']:
        print(f"  - ai4mars_geo: Geology (22 classes)")
    print("\nBoth PNG masks and YOLO text files have been generated!")
    print("You can now use these datasets for YOLO training!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_ai4mars_to_yolo_dual.py /path/to/ai4mars-dataset-merged-0.6 [options]")
        print("\nOptions:")
        print("  --config [nav|geo|both]  Which dataset(s) to create (default: both)")
        print("  --output PATH            Output directory (default: ai4mars-dataset-merged-0.6/AI4MARS_YOLO)")
        print("  --limit N                Process only N files per configuration")
        print("\nExamples:")
        print("  python convert_ai4mars_to_yolo_dual.py /path/to/dataset")
        print("  python convert_ai4mars_to_yolo_dual.py /path/to/dataset --config nav")
        print("  python convert_ai4mars_to_yolo_dual.py /path/to/dataset --config geo")
        print("  python convert_ai4mars_to_yolo_dual.py /path/to/dataset --config both --limit 100")
        sys.exit(1)
    
    dataset_path = Path(sys.argv[1])
    output_path = None
    max_files = None
    config = 'both'
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_path = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--config' and i + 1 < len(sys.argv):
            config = sys.argv[i + 1].lower()
            if config not in ['nav', 'geo', 'both']:
                print(f"Error: --config must be 'nav', 'geo', or 'both'")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--limit' and i + 1 < len(sys.argv):
            try:
                max_files = int(sys.argv[i + 1])
                print(f"Limit set to {max_files} files per configuration\n")
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
    
    process_dataset(dataset_path, output_path, config, max_files)


if __name__ == '__main__':
    main()