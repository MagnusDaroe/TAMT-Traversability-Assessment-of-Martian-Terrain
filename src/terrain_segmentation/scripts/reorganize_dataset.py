"""
Dataset Split and Augmentation Script for YOLO Segmentation Format
Splits existing images/labels into train/valid/test and augments training data.
Handles YOLO polygon format (.txt files).
"""

import os
import shutil
from pathlib import Path
import yaml
import random
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import cv2


def parse_yolo_segmentation(txt_path, img_width, img_height):
    """
    Parse YOLO segmentation format and create mask.
    
    Format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_width)
                y = int(coords[i+1] * img_height)
                points.append((x, y))
            
            # Draw polygon on mask
            img = Image.fromarray(mask)
            draw = ImageDraw.Draw(img)
            if len(points) >= 3:
                draw.polygon(points, fill=class_id)
            mask = np.array(img)
    
    return Image.fromarray(mask)


def mask_to_yolo_segmentation(mask_array, img_width, img_height):
    """
    Convert mask back to YOLO segmentation format.
    Returns list of strings in YOLO format.
    """
    lines = []
    unique_classes = np.unique(mask_array)
    
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
        
        # Create binary mask for this class
        class_mask = (mask_array == class_id).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Simplify contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to normalized coordinates
            coords = []
            for point in approx:
                x = point[0][0] / img_width
                y = point[0][1] / img_height
                # Clamp to [0, 1]
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                coords.append(f"{x:.6f}")
                coords.append(f"{y:.6f}")
            
            if len(coords) >= 6:  # At least 3 points
                line = f"{class_id} " + " ".join(coords)
                lines.append(line)
    
    return lines


def add_noise(image, noise_ratio=0.001):
    """Add random noise to image."""
    img_array = np.array(image)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    num_noise_pixels = int(total_pixels * noise_ratio)
    
    noise_coords = (np.random.randint(0, img_array.shape[0], num_noise_pixels),
                   np.random.randint(0, img_array.shape[1], num_noise_pixels))
    
    if len(img_array.shape) == 3:
        img_array[noise_coords] = np.random.randint(0, 256, (num_noise_pixels, img_array.shape[2]))
    else:
        img_array[noise_coords] = np.random.randint(0, 256, num_noise_pixels)
    
    return Image.fromarray(img_array)


def augment_image(image, mask, aug_type, seed=None):
    """Apply augmentation to both image and mask."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    img_array = np.array(image)
    mask_array = np.array(mask)
    
    if aug_type == 'hflip':
        img_array = np.fliplr(img_array)
        mask_array = np.fliplr(mask_array)
    
    elif aug_type.startswith('rotate_'):
        angle = float(aug_type.split('_')[1])
        h, w = img_array.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        img_array = cv2.warpAffine(img_array, M, (w, h), 
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
        mask_array = cv2.warpAffine(mask_array, M, (w, h), 
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
    
    aug_image = Image.fromarray(img_array)
    aug_mask = Image.fromarray(mask_array)
    
    if aug_type == 'grayscale':
        aug_image = ImageOps.grayscale(aug_image)
        aug_image = aug_image.convert('RGB')
    
    elif aug_type.startswith('hue_'):
        hue_delta = float(aug_type.split('_')[1])
        aug_image = aug_image.convert('HSV')
        h, s, v = aug_image.split()
        
        h_array = np.array(h, dtype=np.int16)
        h_array = (h_array + int(hue_delta * 255 / 360)) % 256
        h = Image.fromarray(h_array.astype(np.uint8), mode='L')
        
        aug_image = Image.merge('HSV', (h, s, v))
        aug_image = aug_image.convert('RGB')
    
    elif aug_type.startswith('noise_'):
        noise_ratio = float(aug_type.split('_')[1])
        aug_image = add_noise(aug_image, noise_ratio)
    
    return aug_image, aug_mask


def generate_augmentation_plan(num_original, target_multiplier=2.0):
    """Generate augmentation plan to reach target multiplier."""
    num_augmented_needed = int(num_original * (target_multiplier - 1))
    augmentations = []
    num_per_type = num_augmented_needed // 5
    remainder = num_augmented_needed % 5
    
    augmentations.extend(['hflip'] * num_per_type)
    
    rotation_angles = np.random.uniform(-15, 15, num_per_type)
    augmentations.extend([f'rotate_{angle:.1f}' for angle in rotation_angles])
    
    num_grayscale = num_per_type // 4
    augmentations.extend(['grayscale'] * num_grayscale)
    
    hue_deltas = np.random.uniform(-15, 15, num_per_type)
    augmentations.extend([f'hue_{delta:.1f}' for delta in hue_deltas])
    
    noise_ratios = np.random.uniform(0.0001, 0.001, num_per_type)
    augmentations.extend([f'noise_{ratio:.6f}' for ratio in noise_ratios])
    
    if remainder > 0:
        extra_augs = random.choices(['hflip', 'grayscale'], k=remainder)
        augmentations.extend(extra_augs)
    
    random.shuffle(augmentations)
    return augmentations


def split_and_augment_dataset(input_dir, output_dir, train_ratio=0.7, valid_ratio=0.15, 
                               test_ratio=0.15, augment_train=True, train_multiplier=2.0, seed=42):
    """Split dataset into train/valid/test and augment training data."""
    random.seed(seed)
    np.random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    images_dir = input_path / 'images'
    labels_dir = input_path / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: {input_dir} must contain 'images' and 'labels' directories")
        return
    
    # Find all image-label pairs
    image_files = sorted([f for f in images_dir.iterdir() 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    pairs = []
    missing_labels = []
    
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        
        if label_file.exists():
            pairs.append((img_file, label_file))
        else:
            missing_labels.append(img_file.name)
    
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images have no corresponding .txt label")
    
    if len(pairs) == 0:
        print("Error: No valid image-label pairs found!")
        return
    
    print(f"Found {len(pairs)} valid image-label pairs")
    
    # Shuffle and split
    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    
    train_pairs = pairs[:n_train]
    valid_pairs = pairs[n_train:n_train+n_valid]
    test_pairs = pairs[n_train+n_valid:]
    
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_pairs)} pairs ({len(train_pairs)/n_total*100:.1f}%)")
    print(f"  Valid: {len(valid_pairs)} pairs ({len(valid_pairs)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} pairs ({len(test_pairs)/n_total*100:.1f}%)")
    
    splits = {'train': train_pairs, 'valid': valid_pairs, 'test': test_pairs}
    split_counts = {}
    
    for split_name, split_pairs in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split...")
        print(f"{'='*60}")
        
        img_out_dir = output_path / 'images' / split_name
        lbl_out_dir = output_path / 'labels' / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy original pairs
        for img_file, lbl_file in split_pairs:
            shutil.copy2(img_file, img_out_dir / img_file.name)
            shutil.copy2(lbl_file, lbl_out_dir / lbl_file.name)
        
        print(f"  ✓ Copied {len(split_pairs)} original pairs")
        
        # Augment training data
        augmented_count = 0
        if augment_train and split_name == 'train':
            print(f"\n  Applying augmentation (target: {train_multiplier}x)...")
            aug_plan = generate_augmentation_plan(len(split_pairs), train_multiplier)
            print(f"  Generated {len(aug_plan)} augmentations")
            
            for i, aug_type in enumerate(aug_plan):
                img_file, lbl_file = random.choice(split_pairs)
                
                try:
                    # Load image
                    image = Image.open(img_file).convert('RGB')
                    img_width, img_height = image.size
                    
                    # Parse YOLO txt to mask
                    mask = parse_yolo_segmentation(lbl_file, img_width, img_height)
                    
                    # Apply augmentation
                    aug_image, aug_mask = augment_image(image, mask, aug_type, seed=i)
                    
                    # Save augmented image
                    aug_img_name = f"{img_file.stem}_aug{i:05d}{img_file.suffix}"
                    aug_image.save(img_out_dir / aug_img_name)
                    
                    # Convert mask back to YOLO format and save
                    aug_mask_array = np.array(aug_mask)
                    yolo_lines = mask_to_yolo_segmentation(aug_mask_array, img_width, img_height)
                    
                    aug_lbl_name = f"{img_file.stem}_aug{i:05d}.txt"
                    with open(lbl_out_dir / aug_lbl_name, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    
                    augmented_count += 1
                    if augmented_count % 100 == 0:
                        print(f"  Generated {augmented_count}/{len(aug_plan)} augmented pairs...")
                        
                except Exception as e:
                    print(f"  Warning: Failed to augment {img_file.name}: {e}")
                    continue
            
            print(f"  ✓ Generated {augmented_count} augmented pairs")
            split_counts[split_name] = len(split_pairs) + augmented_count
            print(f"  ✓ Total: {split_counts[split_name]} pairs ({split_counts[split_name]/len(split_pairs):.1f}x)")
        else:
            split_counts[split_name] = len(split_pairs)
    
    # Create data.yaml
    create_data_yaml(output_path, split_counts, augment_train, train_multiplier)
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset split and augmentation complete!")
    print(f"{'='*60}")
    print(f"  Train: {split_counts['train']} pairs" + 
          (f" ({train_multiplier}x augmented)" if augment_train else ""))
    print(f"  Valid: {split_counts['valid']} pairs")
    print(f"  Test:  {split_counts['test']} pairs")
    print(f"  Total: {sum(split_counts.values())} pairs")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")


def create_data_yaml(output_path, splits, augmented, multiplier):
    """Create data.yaml file."""
    data_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'names': {
            0: 'soil',
            1: 'bedrock',
            2: 'sand',
            3: 'rocks',
            4: 'hole',
        },
        'nc': 6,
        'splits': {
            'train': splits['train'],
            'valid': splits['valid'],
            'test': splits['test']
        }
    }
    
    if augmented:
        data_config['augmentation'] = {
            'applied_to': 'train',
            'multiplier': multiplier,
            'techniques': [
                'horizontal_flip',
                'rotation (-15° to +15°)',
                'grayscale (25% of augmentations)',
                'hue_shift (-15° to +15°)',
                'noise (up to 0.1% of pixels)'
            ]
        }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Created data.yaml")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split dataset and augment training data (YOLO segmentation format)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_dir', help='Input directory (contains images/ and labels/)')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--train', type=float, default=0.7, help='Train ratio (default: 0.7)')
    parser.add_argument('--valid', type=float, default=0.15, help='Valid ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15, help='Test ratio (default: 0.15)')
    parser.add_argument('--no-augment', action='store_true', help='Disable augmentation')
    parser.add_argument('--multiplier', type=float, default=2.0, help='Augmentation multiplier (default: 2.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory does not exist!")
        return
    
    if abs(args.train + args.valid + args.test - 1.0) > 0.01:
        print(f"Error: train + valid + test must equal 1.0")
        return
    
    split_and_augment_dataset(
        args.input_dir, args.output_dir,
        train_ratio=args.train, valid_ratio=args.valid, test_ratio=args.test,
        augment_train=not args.no_augment, train_multiplier=args.multiplier,
        seed=args.seed
    )


if __name__ == '__main__':
    main()