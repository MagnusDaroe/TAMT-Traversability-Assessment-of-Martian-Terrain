"""
YOLO Segmentation Annotation Viewer
Visualize converted YOLO segmentation annotations to verify correctness
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
import random


def load_yolo_segmentation(txt_path, img_width, img_height):
    """
    Load YOLO segmentation annotations from .txt file.
    
    Args:
        txt_path: Path to .txt annotation file
        img_width: Image width for denormalization
        img_height: Image height for denormalization
    
    Returns:
        List of (class_id, polygon_points) tuples
    """
    if not txt_path.exists():
        return []
    
    annotations = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 7:  # class_id + at least 3 points (x,y pairs)
                continue
            
            class_id = int(parts[0])
            
            # Parse polygon coordinates
            coords = [float(x) for x in parts[1:]]
            
            # Convert to pixel coordinates
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_width)
                y = int(coords[i+1] * img_height)
                points.append([x, y])
            
            annotations.append((class_id, np.array(points, dtype=np.int32)))
    
    return annotations


def get_class_colors(num_classes):
    """Generate distinct colors for each class."""
    colors = []
    np.random.seed(42)  # For consistent colors
    for i in range(num_classes):
        color = tuple([int(x) for x in np.random.randint(0, 255, 3)])
        colors.append(color)
    return colors


def visualize_yolo_segmentation(image_path, txt_path, class_names=None, alpha=0.5):
    """
    Visualize YOLO segmentation annotations on an image.
    
    Args:
        image_path: Path to image file
        txt_path: Path to corresponding .txt annotation
        class_names: Dictionary mapping class IDs to names
        alpha: Transparency for mask overlay (0-1)
    
    Returns:
        Annotated image
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    height, width = img.shape[:2]
    
    # Create overlay for masks
    overlay = img.copy()
    
    # Load annotations
    annotations = load_yolo_segmentation(txt_path, width, height)
    
    if not annotations:
        # No annotations, just return original image
        cv2.putText(img, "No annotations", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img
    
    # Get colors for classes
    if class_names:
        num_classes = max(class_names.keys()) + 1
    else:
        num_classes = max([ann[0] for ann in annotations]) + 1
    
    colors = get_class_colors(num_classes)
    
    # Draw each annotation
    for class_id, polygon in annotations:
        color = colors[class_id]
        
        # Draw filled polygon on overlay
        cv2.fillPoly(overlay, [polygon], color)
        
        # Draw polygon outline on original image
        cv2.polylines(img, [polygon], isClosed=True, color=color, thickness=2)
        
        # Add class label
        if len(polygon) > 0:
            # Get centroid for label placement
            M = cv2.moments(polygon)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                label = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (cx - 5, cy - text_height - 5), 
                            (cx + text_width + 5, cy + 5), color, -1)
                
                # Draw label text
                cv2.putText(img, label, (cx, cy), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Blend overlay with original image
    result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    # Add info text
    info_text = f"Annotations: {len(annotations)}"
    cv2.putText(result, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    return result


def view_dataset(dataset_path, split='train', num_samples=10, save_output=False):
    """
    View random samples from the dataset with annotations.
    
    Args:
        dataset_path: Path to dataset directory
        split: Which split to view ('train', 'valid', 'test')
        num_samples: Number of samples to view
        save_output: If True, saves visualizations to disk
    """
    dataset_path = Path(dataset_path)
    
    # Load class names
    class_names = None
    data_yaml = dataset_path / 'data.yaml'
    if data_yaml.exists():
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            class_names = data_config.get('names', {})
    
    print("=" * 60)
    print(f"YOLO SEGMENTATION VIEWER - {split.upper()} SPLIT")
    print("=" * 60)
    if class_names:
        print(f"Classes: {class_names}")
    print("=" * 60)
    
    # Get image and label directories
    images_dir = dataset_path / 'images' / split
    labels_dir = dataset_path / 'labels' / split
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(images_dir.glob(ext)))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"\nFound {len(image_files)} images in {split} split")
    
    # Select random samples
    num_samples = min(num_samples, len(image_files))
    sample_images = random.sample(image_files, num_samples)
    
    print(f"Viewing {num_samples} random samples...")
    print("\nControls:")
    print("  - Press any key to view next image")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current image")
    print("=" * 60)
    
    # Create output directory if saving
    if save_output:
        output_dir = dataset_path / 'visualization_output'
        output_dir.mkdir(exist_ok=True)
        print(f"Saving visualizations to: {output_dir}")
    
    for idx, img_path in enumerate(sample_images):
        print(f"\n[{idx+1}/{num_samples}] {img_path.name}")
        
        # Find corresponding annotation
        txt_path = labels_dir / f"{img_path.stem}.txt"
        
        if not txt_path.exists():
            print(f"  Warning: No annotation file found: {txt_path.name}")
            continue
        
        # Visualize
        result = visualize_yolo_segmentation(img_path, txt_path, class_names)
        
        if result is None:
            continue
        
        # Count annotations
        annotations = load_yolo_segmentation(txt_path, result.shape[1], result.shape[0])
        class_counts = {}
        for class_id, _ in annotations:
            class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"  Annotations: {len(annotations)} segments")
        for class_name, count in sorted(class_counts.items()):
            print(f"    - {class_name}: {count}")
        
        # Resize for display if too large
        display = result.copy()
        max_height = 800
        if display.shape[0] > max_height:
            scale = max_height / display.shape[0]
            new_width = int(display.shape[1] * scale)
            display = cv2.resize(display, (new_width, max_height))
        
        # Show image
        cv2.imshow('YOLO Segmentation Viewer', display)
        
        key = cv2.waitKey(0) & 0xFF
        
        # Save if requested
        if key == ord('s') or save_output:
            output_path = output_dir / f"{img_path.stem}_annotated.jpg" if save_output else f"{img_path.stem}_annotated.jpg"
            cv2.imwrite(str(output_path), result)
            print(f"  ✓ Saved to: {output_path}")
        
        # Quit if 'q' pressed
        if key == ord('q'):
            print("\nQuitting viewer...")
            break
    
    cv2.destroyAllWindows()
    print("\n✓ Viewer closed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='View YOLO segmentation annotations overlaid on images'
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to YOLO format dataset directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='',
        choices=['train', 'valid', 'test'],
        help='Which split to view (default: train)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of random samples to view (default: 10)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save visualizations to disk'
    )
    
    args = parser.parse_args()
    
    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return
    
    view_dataset(args.dataset_path, args.split, args.num_samples, args.save)


if __name__ == '__main__':
    main()