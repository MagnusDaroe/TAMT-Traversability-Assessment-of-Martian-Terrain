#!/usr/bin/env python3
"""
Universal Dataset Splitter for YOLO Format
Works with ANY naming convention - automatically matches images to labels by numeric ID
"""

import os
import shutil
import random
import yaml
import re
from pathlib import Path
from typing import Tuple, List
import argparse


class DatasetSplitter:
    def __init__(self, dataset_path: str, train_split: float = 0.7, 
                 val_split: float = 0.2, test_split: float = 0.1, seed: int = 42):
        self.original_dataset_path = Path(dataset_path)
        parent_dir = self.original_dataset_path.parent
        original_name = self.original_dataset_path.name
        self.dataset_path = parent_dir / f"{original_name}_split"
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        total = train_split + val_split + test_split
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Splits must sum to 1.0, got {total}")
        
        random.seed(seed)
        
    def find_images_and_labels(self) -> List[Tuple[Path, Path]]:
        """Find images and labels - works with any prefix combination."""
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        
        images_dir = self.original_dataset_path / 'images'
        labels_dir = self.original_dataset_path / 'labels'
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory not found: {labels_dir}")
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"  Found {len(image_files)} image files")
        
        # Match images to labels
        valid_pairs = []
        missing_labels = []
        
        for img_path in image_files:
            img_stem = img_path.stem
            
            # Try 1: Exact match (same name)
            exact_label = labels_dir / f"{img_stem}.txt"
            if exact_label.exists():
                valid_pairs.append((img_path, exact_label))
                continue
            
            # Try 2: Match by numeric ID
            id_match = re.search(r'(\d+)$', img_stem)
            if id_match:
                numeric_id = id_match.group(1)
                possible_labels = list(labels_dir.glob(f"*{numeric_id}.txt"))
                
                if possible_labels:
                    valid_pairs.append((img_path, possible_labels[0]))
                    continue
            
            missing_labels.append(img_path)
        
        if missing_labels:
            print(f"  ⚠️  {len(missing_labels)} images missing labels")
            if len(missing_labels) <= 5:
                for img in missing_labels[:5]:
                    print(f"     - {img.name}")
        
        if not valid_pairs:
            raise ValueError("No valid image-label pairs found")
        
        print(f"  ✓ Matched {len(valid_pairs)} image-label pairs")
        if valid_pairs:
            ex_img, ex_label = valid_pairs[0]
            print(f"  Example: {ex_img.name} → {ex_label.name}")
        
        return valid_pairs
    
    def split_data(self, pairs):
        pairs_copy = pairs.copy()
        random.shuffle(pairs_copy)
        
        total = len(pairs_copy)
        train_end = int(total * self.train_split)
        val_end = train_end + int(total * self.val_split)
        
        train_pairs = pairs_copy[:train_end]
        val_pairs = pairs_copy[train_end:val_end]
        test_pairs = pairs_copy[val_end:]
        
        print(f"\n📊 Split Statistics:")
        print(f"   Train: {len(train_pairs)} ({len(train_pairs)/total*100:.1f}%)")
        print(f"   Val:   {len(val_pairs)} ({len(val_pairs)/total*100:.1f}%)")
        print(f"   Test:  {len(test_pairs)} ({len(test_pairs)/total*100:.1f}%)")
        
        return train_pairs, val_pairs, test_pairs
    
    def create_split_dataset_directory(self):
        if self.dataset_path.exists():
            print(f"⚠️  Split dataset exists at {self.dataset_path}")
            response = input("Overwrite? (yes/no): ")
            if response.lower() != 'yes':
                raise ValueError("Cancelled")
            shutil.rmtree(self.dataset_path)
        
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        original_yaml = self.original_dataset_path / 'data.yaml'
        if original_yaml.exists():
            shutil.copy2(original_yaml, self.dataset_path / 'data.yaml')
        
        print(f"✓ Created: {self.dataset_path}")
    
    def create_split_directories(self):
        for split in ['train', 'val', 'test']:
            (self.dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        print("✓ Created split directories")
    
    def copy_files(self, pairs, split_name):
        """Copy and rename labels to match image names."""
        for img_path, label_path in pairs:
            # Copy image with original name
            dest_img = self.dataset_path / 'images' / split_name / img_path.name
            
            # Copy label with NEW name matching image stem
            label_name_new = f"{img_path.stem}.txt"
            dest_label = self.dataset_path / 'labels' / split_name / label_name_new
            
            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_path, dest_label)
        
        print(f"✓ Copied {len(pairs)} to {split_name} (labels renamed)")
    
    def update_data_yaml(self):
        yaml_path = self.dataset_path / 'data.yaml'
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            data = {}
        
        data['path'] = str(self.dataset_path.resolve())
        data['train'] = 'images/train'
        data['val'] = 'images/val'
        data['test'] = 'images/test'
        
        if 'nc' not in data:
            data['nc'] = 6
        
        if 'names' not in data:
            data['names'] = {
                0: 'BACKGROUND',
                1: 'UNLABELLED', 
                2: 'bedrock',
                3: 'loose sand',
                4: 'rocks',
                5: 'soil'
            }
        elif isinstance(data['names'], list):
            data['names'] = {i: name for i, name in enumerate(data['names'])}
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Updated data.yaml")
    
    def split(self):
        print(f"\n{'='*60}")
        print("Dataset Splitter")
        print(f"{'='*60}")
        print(f"Original: {self.original_dataset_path}")
        print(f"Split to: {self.dataset_path}")
        print(f"{'='*60}\n")
        
        print("Step 1: Finding image-label pairs...")
        pairs = self.find_images_and_labels()
        
        print("\nStep 2: Creating split dataset...")
        self.create_split_dataset_directory()
        
        print("\nStep 3: Splitting data...")
        train_pairs, val_pairs, test_pairs = self.split_data(pairs)
        
        print("\nStep 4: Creating directories...")
        self.create_split_directories()
        
        print("\nStep 5: Copying files...")
        self.copy_files(train_pairs, 'train')
        self.copy_files(val_pairs, 'val')
        self.copy_files(test_pairs, 'test')
        
        print("\nStep 6: Updating data.yaml...")
        self.update_data_yaml()
        
        print(f"\n{'='*60}")
        print("✅ Complete!")
        print(f"{'='*60}")
        print(f"\nDataset ready: {self.dataset_path}")
        print(f"  Train: {len(train_pairs)} images")
        print(f"  Val:   {len(val_pairs)} images")
        print(f"  Test:  {len(test_pairs)} images\n")


def main():
    parser = argparse.ArgumentParser(description='Universal YOLO Dataset Splitter')
    parser.add_argument('dataset_path', help='Dataset root directory')
    parser.add_argument('--train', type=float, default=0.7)
    parser.add_argument('--val', type=float, default=0.2)
    parser.add_argument('--test', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    try:
        splitter = DatasetSplitter(
            dataset_path=args.dataset_path,
            train_split=args.train,
            val_split=args.val,
            test_split=args.test,
            seed=args.seed
        )
        splitter.split()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())