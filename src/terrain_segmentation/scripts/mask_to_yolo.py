"""
Convert PNG segmentation masks to YOLO segmentation format (.txt files)
YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ... (normalized polygon coordinates)
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import shutil


def mask_to_yolo_segmentation(mask, class_id, img_width, img_height, min_area=10):
    """
    Convert a binary mask for a single class to YOLO segmentation format.
    
    Args:
        mask: Binary mask for one class (numpy array, uint8)
        class_id: Class ID for this mask
        img_width: Image width for normalization
        img_height: Image height for normalization
        min_area: Minimum contour area to consider (filters noise)
    
    Returns:
        List of YOLO format strings (one per contour/object instance)
    """
    # Find contours for this class
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_segments = []
    
    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Skip tiny contours (likely noise)
        if area < min_area:
            continue
        
        # Simplify contour to reduce points (optional, helps reduce file size)
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Need at least 3 points for a polygon
        if len(contour) < 3:
            continue
        
        # Flatten and normalize coordinates
        segment = []
        for point in contour:
            x, y = point[0]
            # Normalize to 0-1 range
            x_norm = x / img_width
            y_norm = y / img_height
            # Clip to valid range
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            segment.extend([x_norm, y_norm])
        
        # Create YOLO format string: class_id x1 y1 x2 y2 x3 y3 ...
        yolo_line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in segment)
        yolo_segments.append(yolo_line)
    
    return yolo_segments


def convert_mask_to_yolo_txt(mask_path, output_txt_path, class_names=None):
    """
    Convert a single PNG mask to YOLO segmentation .txt file.
    
    Args:
        mask_path: Path to the PNG mask file
        output_txt_path: Path where to save the .txt file
        class_names: Optional dict mapping class IDs to names (for logging)
    
    Returns:
        Number of segments written
    """
    # Read mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Warning: Could not read {mask_path}")
        return 0
    
    height, width = mask.shape
    
    # Get unique class IDs in the mask (excluding 0 which is often background)
    unique_classes = np.unique(mask)
    
    all_segments = []
    
    # Process each class
    for class_id in unique_classes:
        # Skip background (class 0) if you want
        # Uncomment the next line if background should be excluded:
        # if class_id == 0:
        #     continue
        
        # Create binary mask for this class
        class_mask = (mask == class_id).astype(np.uint8)
        
        # Convert to YOLO segments
        segments = mask_to_yolo_segmentation(class_mask, class_id, width, height)
        all_segments.extend(segments)
    
    # Write to file
    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(all_segments))
        if all_segments:  # Add final newline if there's content
            f.write('\n')
    
    return len(all_segments)


def convert_dataset_masks_to_yolo(dataset_path, output_path=None, skip_background=False):
    """
    Convert all PNG masks in a dataset to YOLO segmentation format.
    
    Args:
        dataset_path: Path to dataset with images/ and labels/ folders
        output_path: Output path (if None, creates new folder with '_yolo_format' suffix)
        skip_background: If True, skips class 0 (background)
    """
    dataset_path = Path(dataset_path)
    
    if output_path is None:
        output_path = dataset_path.parent / f"{dataset_path.name}_yolo_format"
    else:
        output_path = Path(output_path)
    
    print("=" * 60)
    print("PNG MASK TO YOLO SEGMENTATION CONVERSION")
    print("=" * 60)
    print(f"Input:  {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Skip background (class 0): {skip_background}")
    print("=" * 60)
    
    # Load class names from data.yaml if available
    class_names = None
    data_yaml = dataset_path / 'data.yaml'
    if data_yaml.exists():
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            class_names = data_config.get('names', {})
            print(f"\nClass names: {class_names}")
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"\nSkipping {split} (directory not found)")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Processing {split.upper()} split")
        print(f"{'=' * 60}")
        
        # Create output directories
        out_images_dir = output_path / 'images' / split
        out_labels_dir = output_path / 'labels' / split
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all mask files
        mask_files = list(labels_dir.glob('*.png'))
        
        if not mask_files:
            print(f"No PNG masks found in {labels_dir}")
            continue
        
        print(f"Found {len(mask_files)} mask files")
        
        total_segments = 0
        converted_count = 0
        empty_count = 0
        
        # Process each mask
        for mask_file in tqdm(mask_files, desc=f"Converting {split}"):
            # Output .txt file (same name as mask, but .txt extension)
            txt_file = out_labels_dir / f"{mask_file.stem}.txt"
            
            # Convert mask to YOLO format
            num_segments = convert_mask_to_yolo_txt(mask_file, txt_file, class_names)
            
            total_segments += num_segments
            if num_segments > 0:
                converted_count += 1
            else:
                empty_count += 1
            
            # Copy corresponding image
            # Try different image extensions
            image_copied = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_file = images_dir / f"{mask_file.stem}{ext}"
                if img_file.exists():
                    shutil.copy2(img_file, out_images_dir / img_file.name)
                    image_copied = True
                    break
            
            if not image_copied:
                print(f"\nWarning: No image found for {mask_file.name}")
        
        print(f"\n{split.upper()} Results:")
        print(f"  Converted: {converted_count} files")
        print(f"  Empty (no objects): {empty_count} files")
        print(f"  Total segments: {total_segments}")
    
    # Copy and update data.yaml
    print(f"\n{'=' * 60}")
    print("Creating data.yaml")
    print(f"{'=' * 60}")
    
    if data_yaml.exists():
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update paths
        data_config['path'] = str(output_path.absolute())
        
        # Remove 'masks' section since we're using standard YOLO format now
        if 'masks' in data_config:
            del data_config['masks']
        
        # Write new data.yaml
        output_yaml = output_path / 'data.yaml'
        with open(output_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Created {output_yaml}")
    else:
        print("⚠️  No data.yaml found in source, you'll need to create one manually")
    
    print(f"\n{'=' * 60}")
    print("CONVERSION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Your YOLO-format dataset is ready at: {output_path}")
    print("\nYou can now train with:")
    print(f"  yolo segment train model=yolo11m-seg.pt data={output_path}/data.yaml")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert PNG masks to YOLO segmentation format (.txt with polygons)'
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to dataset directory (containing images/ and labels/ folders)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: input_dir + "_yolo_format")'
    )
    parser.add_argument(
        '--skip-background',
        action='store_true',
        help='Skip class 0 (background) in conversion'
    )
    
    args = parser.parse_args()
    
    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return
    
    convert_dataset_masks_to_yolo(
        args.dataset_path,
        args.output,
        args.skip_background
    )


if __name__ == '__main__':
    main()