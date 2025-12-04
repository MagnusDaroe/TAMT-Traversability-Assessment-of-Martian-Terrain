"""
Simple converter: Mask images to YOLO txt format.
Creates images/ and labels/ folders with corresponding files.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


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
        
        # Normalize coordinates
        normalized = contour.astype(float)
        normalized[:, 0] /= width
        normalized[:, 1] /= height
        
        # Clip to valid range [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)
        
        segment = [class_id] + normalized.flatten().tolist()
        segments.append(segment)
    
    return segments


def convert_mask_to_yolo_txt(mask_path, output_txt_path):
    """
    Convert a mask image to YOLO text format.
    Remaps classes from mask values to YOLO IDs:
    - Mask 1 (bedrock) -> YOLO 1
    - Mask 2 (hole) -> YOLO 4
    - Mask 3 (rocks) -> YOLO 3
    - Mask 4 (sand) -> YOLO 2
    - Mask 5 (soil) -> YOLO 0
    """
    # Mapping from mask pixel value to YOLO class ID
    class_mapping = {
        1: 1,  # bedrock -> 1
        2: 4,  # hole -> 4
        3: 3,  # rocks -> 3
        4: 2,  # sand -> 2
        5: 0,  # soil -> 0
    }
    
    # Read mask as grayscale
    class_label = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if class_label is None:
        return False
    
    # Get unique class IDs (exclude background 0 and 255)
    unique_classes = np.unique(class_label)
    unique_classes = unique_classes[(unique_classes > 0) & (unique_classes < 255)]
    
    all_segments = []
    
    for class_id in unique_classes:
        # Skip if class not in mapping
        if int(class_id) not in class_mapping:
            continue
            
        # Create binary mask for this class
        class_mask = (class_label == class_id).astype(np.uint8)
        
        # Get the remapped YOLO class ID
        yolo_class_id = class_mapping[int(class_id)]
        
        # Convert to YOLO segments
        segments = mask_to_yolo_segments(class_mask, yolo_class_id)
        all_segments.extend(segments)
    
    # Write to file
    with open(output_txt_path, 'w') as f:
        for segment in all_segments:
            line = ' '.join(map(str, segment))
            f.write(line + '\n')
    
    return len(all_segments) > 0


def convert_dataset(input_dir, output_dir):
    """
    Convert mask dataset to YOLO format.
    
    Args:
        input_dir: Directory containing image and mask files
        output_dir: Output directory (will create images/ and labels/ subdirs)
    
    Class mapping:
        0: soil
        1: bedrock
        2: sand
        3: rocks
        4: hole
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files (not masks)
    all_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg')) + list(input_path.glob('*.png'))
    image_files = [f for f in all_files if '_mask' not in f.name]
    
    print(f"\nFound {len(image_files)} images")
    
    converted = 0
    failed = 0
    
    # Process each image
    for img_file in tqdm(image_files, desc="Converting"):
        # Find corresponding mask
        base_name = img_file.stem
        mask_file = input_path / f"{base_name}_mask.png"
        
        if not mask_file.exists():
            # Try other extensions
            mask_file = input_path / f"{base_name}_mask.jpg"
            if not mask_file.exists():
                failed += 1
                continue
        
        try:
            # Copy RGB image
            img = cv2.imread(str(img_file))
            if img is None:
                failed += 1
                continue
            
            # Ensure RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Save RGB image
            output_img = images_dir / img_file.name
            cv2.imwrite(str(output_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Convert mask to txt
            output_txt = labels_dir / f"{img_file.stem}.txt"
            if convert_mask_to_yolo_txt(mask_file, output_txt):
                converted += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"\nError processing {img_file.name}: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Conversion Complete")
    print(f"{'='*50}")
    print(f"✓ Converted: {converted}")
    print(f"✗ Failed: {failed}")
    print(f"\nOutput:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert mask images to YOLO txt format. Class mapping: 0=soil, 1=bedrock, 2=sand, 3=rocks, 4=hole'
    )
    parser.add_argument('input_dir', help='Input directory with images and masks')
    parser.add_argument('output_dir', help='Output directory')
    
    args = parser.parse_args()
    
    if not Path(args.input_dir).exists():
        print(f"Error: {args.input_dir} does not exist")
        return
    
    print("Class mapping:")
    print("  0: soil")
    print("  1: bedrock")
    print("  2: sand")
    print("  3: rocks")
    print("  4: hole")
    print()
    
    convert_dataset(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()