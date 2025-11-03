#!/usr/bin/env python3
"""
Dataset Validator - Check for missing or empty labels in YOLO dataset
"""

import os
import yaml
from pathlib import Path
import argparse


def validate_yolo_dataset(dataset_path):
    """Validate YOLO dataset structure and label files"""
    
    dataset_path = Path(dataset_path).expanduser()
    
    print(f"\n{'='*70}")
    print(f"YOLO Dataset Validator")
    print(f"{'='*70}")
    print(f"Dataset path: {dataset_path}\n")
    
    # Check if data.yaml exists
    data_yaml = dataset_path / 'data.yaml'
    if not data_yaml.exists():
        print(f"❌ ERROR: data.yaml not found at {data_yaml}")
        return
    
    print(f"✓ Found data.yaml")
    
    # Load data.yaml
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nDataset Configuration:")
    print(f"  Path: {config.get('path', 'NOT SET')}")
    print(f"  Classes: {config.get('nc', 'NOT SET')}")
    print(f"  Names: {config.get('names', 'NOT SET')}")
    
    # Check each split
    splits_to_check = ['train', 'val', 'test']
    
    for split in splits_to_check:
        if split not in config:
            print(f"\n⚠️  Split '{split}' not defined in data.yaml")
            continue
        
        print(f"\n{'='*70}")
        print(f"Checking {split.upper()} split...")
        print(f"{'='*70}")
        
        # Get paths
        split_path = config[split]
        if not os.path.isabs(split_path):
            base_path = config.get('path', str(dataset_path))
            if not os.path.isabs(base_path):
                base_path = str(dataset_path)
            split_path = os.path.join(base_path, split_path)
        
        split_path = Path(split_path).expanduser()
        
        print(f"Images path: {split_path}")
        
        # Get labels path (replace 'images' with 'labels')
        labels_path = Path(str(split_path).replace('/images/', '/labels/'))
        print(f"Labels path: {labels_path}")
        
        # Check if directories exist
        if not split_path.exists():
            print(f"❌ ERROR: Images directory not found: {split_path}")
            continue
        
        if not labels_path.exists():
            print(f"❌ ERROR: Labels directory not found: {labels_path}")
            continue
        
        # Get all images
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(split_path.glob(f"*{ext}")))
            image_files.extend(list(split_path.glob(f"*{ext.upper()}")))
        
        print(f"\n📊 Statistics:")
        print(f"  Total images: {len(image_files)}")
        
        if len(image_files) == 0:
            print(f"❌ ERROR: No images found in {split_path}")
            continue
        
        # Check labels
        missing_labels = []
        empty_labels = []
        valid_labels = []
        label_instance_counts = []
        
        for img_path in image_files:
            # Handle rgb_ prefix if present
            img_stem = img_path.stem
            if img_stem.startswith('rgb_'):
                label_name = f"yolo_seg_{img_stem[4:]}.txt"
            else:
                label_name = f"{img_stem}.txt"
            
            label_path = labels_path / label_name
            
            if not label_path.exists():
                missing_labels.append(img_path.name)
            else:
                # Check if label is empty
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    lines = [l.strip() for l in lines if l.strip()]  # Remove empty lines
                
                if len(lines) == 0:
                    empty_labels.append(img_path.name)
                else:
                    valid_labels.append(img_path.name)
                    label_instance_counts.append(len(lines))
        
        # Print results
        print(f"  Valid labels: {len(valid_labels)}")
        print(f"  Empty labels: {len(empty_labels)}")
        print(f"  Missing labels: {len(missing_labels)}")
        
        if len(valid_labels) > 0:
            avg_instances = sum(label_instance_counts) / len(label_instance_counts)
            total_instances = sum(label_instance_counts)
            print(f"  Total instances: {total_instances}")
            print(f"  Avg instances per image: {avg_instances:.2f}")
        
        # Show issues
        if missing_labels:
            print(f"\n⚠️  WARNING: {len(missing_labels)} images have no label file")
            if len(missing_labels) <= 10:
                print("  Missing labels for:")
                for img in missing_labels[:10]:
                    print(f"    - {img}")
            else:
                print(f"  (showing first 10)")
                for img in missing_labels[:10]:
                    print(f"    - {img}")
        
        if empty_labels:
            print(f"\n⚠️  WARNING: {len(empty_labels)} label files are EMPTY")
            if len(empty_labels) <= 10:
                print("  Empty labels for:")
                for img in empty_labels[:10]:
                    print(f"    - {img}")
            else:
                print(f"  (showing first 10)")
                for img in empty_labels[:10]:
                    print(f"    - {img}")
        
        # Overall status
        print(f"\n{'─'*70}")
        if len(valid_labels) == 0:
            print(f"❌ CRITICAL: NO VALID LABELS found in {split} split!")
            print(f"   Training/validation will FAIL for this split.")
        elif len(valid_labels) < len(image_files) * 0.5:
            print(f"⚠️  WARNING: Less than 50% of images have valid labels")
            print(f"   This will significantly impact training quality.")
        else:
            print(f"✓ Split {split} looks OK")
        
        # Sample a few labels to check format
        if len(valid_labels) > 0:
            print(f"\n📄 Sample label content (first valid label):")
            sample_img = None
            for img_path in image_files:
                img_stem = img_path.stem
                if img_stem.startswith('rgb_'):
                    label_name = f"yolo_seg_{img_stem[4:]}.txt"
                else:
                    label_name = f"{img_stem}.txt"
                
                label_path = labels_path / label_name
                if label_path.exists():
                    sample_img = img_path
                    break
            
            if sample_img:
                img_stem = sample_img.stem
                if img_stem.startswith('rgb_'):
                    label_name = f"yolo_seg_{img_stem[4:]}.txt"
                else:
                    label_name = f"{img_stem}.txt"
                
                label_path = labels_path / label_name
                print(f"  File: {label_path.name}")
                with open(label_path, 'r') as f:
                    lines = f.readlines()[:5]  # First 5 lines
                    for i, line in enumerate(lines, 1):
                        print(f"    Line {i}: {line.strip()}")
                    if len(lines) == 0:
                        print("    (empty file)")
    
    print(f"\n{'='*70}")
    print("Validation Complete")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Validate YOLO dataset and check for missing/empty labels'
    )
    parser.add_argument('dataset_path', type=str,
                        help='Path to the dataset root directory')
    
    args = parser.parse_args()
    validate_yolo_dataset(args.dataset_path)


if __name__ == '__main__':
    main()