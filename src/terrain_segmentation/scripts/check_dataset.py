#!/usr/bin/env python3
"""
Dataset Diagnostic Tool
Analyzes a dataset to find all images and identify issues
"""

import os
from pathlib import Path
from collections import defaultdict
import argparse


def analyze_dataset(dataset_path):
    """Analyze the dataset structure and count files."""
    dataset_path = Path(dataset_path)
    
    print(f"\n{'='*70}")
    print(f"Dataset Analysis: {dataset_path}")
    print(f"{'='*70}\n")
    
    # Check if path exists
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    print(f"✓ Dataset path exists\n")
    
    # Find all subdirectories
    print("📁 Directory Structure:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(str(dataset_path), '').count(os.sep)
        indent = ' ' * 2 * level
        rel_path = Path(root).relative_to(dataset_path) if root != str(dataset_path) else Path('.')
        print(f'{indent}{rel_path}/')
        if level < 2:  # Only show files for first 2 levels
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... and {len(files) - 5} more files')
    
    # Image extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    label_extension = '.txt'
    
    # Count files by type
    images_by_location = defaultdict(list)
    labels_by_location = defaultdict(list)
    other_files = defaultdict(list)
    
    print(f"\n{'='*70}")
    print("📊 File Analysis:")
    print(f"{'='*70}\n")
    
    for root, dirs, files in os.walk(dataset_path):
        rel_root = Path(root).relative_to(dataset_path)
        
        for file in files:
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            
            if ext in image_extensions:
                images_by_location[str(rel_root)].append(file)
            elif ext == label_extension:
                labels_by_location[str(rel_root)].append(file)
            else:
                other_files[str(rel_root)].append(file)
    
    # Print image statistics
    total_images = sum(len(imgs) for imgs in images_by_location.values())
    print(f"🖼️  Total Images Found: {total_images}")
    
    if images_by_location:
        print("\n   Images by location:")
        for location, images in sorted(images_by_location.items()):
            print(f"      {location}: {len(images)} images")
            # Show sample filenames
            if len(images) > 0:
                print(f"         Sample: {images[0]}")
    
    # Print label statistics
    total_labels = sum(len(lbls) for lbls in labels_by_location.values())
    print(f"\n🏷️  Total Labels Found: {total_labels}")
    
    if labels_by_location:
        print("\n   Labels by location:")
        for location, labels in sorted(labels_by_location.items()):
            print(f"      {location}: {len(labels)} labels")
    
    # Check for common issues
    print(f"\n{'='*70}")
    print("⚠️  Potential Issues:")
    print(f"{'='*70}\n")
    
    issues_found = False
    
    # Issue 1: Images and labels in different locations
    image_locations = set(images_by_location.keys())
    label_locations = set(labels_by_location.keys())
    
    if image_locations != label_locations:
        issues_found = True
        print("⚠️  Images and labels are in different directories:")
        
        images_only = image_locations - label_locations
        if images_only:
            print(f"\n   Directories with images but no labels:")
            for loc in images_only:
                print(f"      - {loc} ({len(images_by_location[loc])} images)")
        
        labels_only = label_locations - image_locations
        if labels_only:
            print(f"\n   Directories with labels but no images:")
            for loc in labels_only:
                print(f"      - {loc} ({len(labels_by_location[loc])} labels)")
    
    # Issue 2: Missing labels for images
    for location in image_locations:
        images = images_by_location[location]
        labels = labels_by_location.get(location, [])
        
        # Create sets of filenames without extensions
        image_names = {Path(img).stem for img in images}
        label_names = {Path(lbl).stem for lbl in labels}
        
        missing_labels = image_names - label_names
        if missing_labels:
            issues_found = True
            print(f"\n⚠️  In '{location}': {len(missing_labels)} images missing labels")
            if len(missing_labels) <= 10:
                for name in sorted(list(missing_labels)[:10]):
                    print(f"      - {name}")
    
    # Issue 3: Check if images/labels directories exist
    expected_images_dir = dataset_path / 'images'
    expected_labels_dir = dataset_path / 'labels'
    
    if not expected_images_dir.exists():
        issues_found = True
        print(f"\n⚠️  Expected 'images' directory not found: {expected_images_dir}")
        print("   Suggestion: Your dataset might have a different structure")
    
    if not expected_labels_dir.exists():
        issues_found = True
        print(f"\n⚠️  Expected 'labels' directory not found: {expected_labels_dir}")
        print("   Suggestion: Your dataset might have a different structure")
    
    # Issue 4: Images scattered in multiple subdirectories
    if len(images_by_location) > 1:
        print(f"\n⚠️  Images found in {len(images_by_location)} different locations")
        print("   The split script expects images in a single 'images' directory")
        print("   It will process ALL images recursively, but this might not be what you want")
    
    if not issues_found:
        print("✓ No obvious issues detected!")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("💡 Recommendations:")
    print(f"{'='*70}\n")
    
    if total_images < 10000:
        print(f"⚠️  You have {total_images} images, but expected ~26,000")
        print("   Possible reasons:")
        print("   1. Images are in subdirectories not being scanned")
        print("   2. Dataset is incomplete or partially downloaded")
        print("   3. Images have unexpected file extensions")
        print("   4. Wrong dataset directory specified")
        print("\n   Try:")
        print(f"      find {dataset_path} -type f -name '*.jpg' | wc -l")
        print(f"      find {dataset_path} -type f -name '*.png' | wc -l")
    
    if total_images != total_labels:
        print(f"\n⚠️  Mismatch: {total_images} images vs {total_labels} labels")
        print("   Some images may be skipped during training")
    
    # Show expected structure
    print("\n📋 Expected YOLO dataset structure:")
    print("""
    dataset/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── labels/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── data.yaml
    """)
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze dataset structure and identify issues',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('dataset_path', type=str,
                        help='Path to the dataset root directory')
    
    args = parser.parse_args()
    
    try:
        analyze_dataset(args.dataset_path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())