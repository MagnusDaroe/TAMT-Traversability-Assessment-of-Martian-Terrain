#!/usr/bin/env python3
"""Debug script to show actual filenames from the dataset."""

from pathlib import Path
import sys

def debug_dataset(root_dir):
    root_dir = Path(root_dir)
    
    print("=" * 70)
    print("LABEL FILES (first 5):")
    print("=" * 70)
    
    labels = []
    for mission in ['m2020', 'mer', 'msl']:
        mission_path = root_dir / mission
        if not mission_path.exists():
            continue
        
        for label_path in mission_path.rglob('*.png'):
            if 'labels' in str(label_path):
                labels.append(label_path)
    
    for label in labels[:5]:
        print(f"\nLabel path: {label.relative_to(root_dir)}")
        print(f"  Filename: {label.name}")
        print(f"  Stem: {label.stem}")
        print(f"  Parent: {label.parent.name}")
    
    print("\n" + "=" * 70)
    print("IMAGE FILES (first 5 from each mission):")
    print("=" * 70)
    
    image_exts = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    
    for mission in ['m2020', 'mer', 'msl']:
        mission_path = root_dir / mission
        if not mission_path.exists():
            continue
        
        print(f"\n{mission.upper()}:")
        
        images = []
        for img_path in mission_path.rglob('*'):
            if img_path.suffix in image_exts and img_path.is_file():
                # Skip if it's in labels directory
                if 'labels' not in str(img_path):
                    images.append(img_path)
        
        for img in images[:5]:
            print(f"  Image: {img.relative_to(root_dir)}")
            print(f"    Filename: {img.name}")
            print(f"    Stem: {img.stem}")
    
    # Now try to find a match
    print("\n" + "=" * 70)
    print("TRYING TO MATCH:")
    print("=" * 70)
    
    if labels:
        test_label = labels[0]
        print(f"\nTest label: {test_label.relative_to(root_dir)}")
        print(f"  Looking for image with stem: '{test_label.stem}'")
        
        # Search for matching image
        mission = test_label.relative_to(root_dir).parts[0]
        mission_path = root_dir / mission
        
        print(f"\n  Searching in mission: {mission}")
        
        # Find all images in this mission
        for img_path in mission_path.rglob('*'):
            if img_path.suffix in image_exts and img_path.is_file():
                if 'labels' not in str(img_path):
                    if img_path.stem == test_label.stem:
                        print(f"  ✓ FOUND MATCH: {img_path.relative_to(root_dir)}")
                        break
        else:
            print(f"  ✗ No match found")
            print(f"\n  Let me show some image stems from this mission:")
            img_count = 0
            for img_path in mission_path.rglob('*'):
                if img_path.suffix in image_exts and img_path.is_file():
                    if 'labels' not in str(img_path):
                        print(f"    Image stem: '{img_path.stem}'")
                        img_count += 1
                        if img_count >= 5:
                            break

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_dataset.py /path/to/ai4mars-dataset-merged-0.6")
        sys.exit(1)
    
    debug_dataset(sys.argv[1])