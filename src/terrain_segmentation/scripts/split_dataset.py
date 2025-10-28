#!/usr/bin/env python3
"""
Dataset Splitter for YOLO Format
Splits images and labels into train/val/test sets while maintaining the directory structure.
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from typing import Tuple, List
import argparse


class DatasetSplitter:
    def __init__(self, dataset_path: str, train_split: float = 0.7, 
                 val_split: float = 0.2, test_split: float = 0.1, seed: int = 42):
        """
        Initialize the dataset splitter.
        
        Args:
            dataset_path: Path to the dataset root directory
            train_split: Percentage of data for training (0.0-1.0)
            val_split: Percentage of data for validation (0.0-1.0)
            test_split: Percentage of data for testing (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.original_dataset_path = Path(dataset_path)
        
        # Create the new split dataset path
        parent_dir = self.original_dataset_path.parent
        original_name = self.original_dataset_path.name
        self.dataset_path = parent_dir / f"{original_name}_split"
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        # Validate splits
        total = train_split + val_split + test_split
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Splits must sum to 1.0, got {total}")
        
        # Set random seed
        random.seed(seed)
        
    def find_images_and_labels(self) -> Tuple[List[Path], List[Path]]:
        """Find all images and their corresponding labels in the dataset."""
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # Look for images directory in the ORIGINAL dataset
        images_dir = self.original_dataset_path / 'images'
        labels_dir = self.original_dataset_path / 'labels'
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory not found: {labels_dir}")
        
        # Find all image files (recursively to handle subdirectories)
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.rglob(f"*{ext}"))
        
        if not image_files:
            raise ValueError(f"No images found in {images_dir}")
        
        # Find corresponding label files
        valid_pairs = []
        missing_labels = []
        
        for img_path in image_files:
            # Get relative path from images directory
            rel_path = img_path.relative_to(images_dir)
            
            # Construct label path (change extension to .txt)
            label_path = labels_dir / rel_path.with_suffix('.txt')
            
            if label_path.exists():
                valid_pairs.append((img_path, label_path))
            else:
                missing_labels.append(img_path)
        
        if missing_labels:
            print(f"⚠️  Warning: {len(missing_labels)} images have no corresponding labels")
            if len(missing_labels) <= 10:
                for img in missing_labels:
                    print(f"   - {img.name}")
        
        if not valid_pairs:
            raise ValueError("No valid image-label pairs found")
        
        print(f"✓ Found {len(valid_pairs)} valid image-label pairs")
        return valid_pairs
    
    def split_data(self, pairs: List[Tuple[Path, Path]]) -> Tuple[List, List, List]:
        """Split the data into train/val/test sets."""
        # Shuffle the pairs
        pairs_copy = pairs.copy()
        random.shuffle(pairs_copy)
        
        total = len(pairs_copy)
        train_end = int(total * self.train_split)
        val_end = train_end + int(total * self.val_split)
        
        train_pairs = pairs_copy[:train_end]
        val_pairs = pairs_copy[train_end:val_end]
        test_pairs = pairs_copy[val_end:]
        
        print(f"\n📊 Split Statistics:")
        print(f"   Train: {len(train_pairs)} images ({len(train_pairs)/total*100:.1f}%)")
        print(f"   Val:   {len(val_pairs)} images ({len(val_pairs)/total*100:.1f}%)")
        print(f"   Test:  {len(test_pairs)} images ({len(test_pairs)/total*100:.1f}%)")
        
        return train_pairs, val_pairs, test_pairs
    
    def create_split_dataset_directory(self):
        """Create the new split dataset directory structure."""
        if self.dataset_path.exists():
            print(f"⚠️  Warning: Split dataset already exists at {self.dataset_path}")
            response = input("Do you want to overwrite it? (yes/no): ")
            if response.lower() != 'yes':
                raise ValueError("Operation cancelled by user")
            shutil.rmtree(self.dataset_path)
        
        # Create the base directory
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Copy data.yaml if it exists in the original
        original_yaml = self.original_dataset_path / 'data.yaml'
        if original_yaml.exists():
            shutil.copy2(original_yaml, self.dataset_path / 'data.yaml')
        
        print(f"✓ Created split dataset directory: {self.dataset_path}")
    
    def create_split_directories(self):
        """Create the train/val/test directory structure."""
        for split in ['train', 'val', 'test']:
            (self.dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        print("✓ Created split directories")
    
    def copy_files(self, pairs: List[Tuple[Path, Path]], split_name: str):
        """Copy image-label pairs to the appropriate split directory."""
        for img_path, label_path in pairs:
            # Destination paths
            dest_img = self.dataset_path / 'images' / split_name / img_path.name
            dest_label = self.dataset_path / 'labels' / split_name / label_path.name
            
            # Copy files
            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_path, dest_label)
        
        print(f"✓ Copied {len(pairs)} files to {split_name} split")
    
    def backup_original_data(self):
        """Backup original images and labels directories."""
        images_dir = self.dataset_path / 'images'
        labels_dir = self.dataset_path / 'labels'
        
        backup_images = self.dataset_path / 'images_original_backup'
        backup_labels = self.dataset_path / 'labels_original_backup'
        
        # Only backup if not already backed up
        if not backup_images.exists() and not backup_labels.exists():
            print("📦 Creating backup of original data...")
            shutil.copytree(images_dir, backup_images)
            shutil.copytree(labels_dir, backup_labels)
            print("✓ Backup created")
        else:
            print("ℹ️  Backup already exists, skipping")
    
    def update_data_yaml(self):
        """Update or create the data.yaml file with correct paths."""
        yaml_path = self.dataset_path / 'data.yaml'
        
        # Read existing yaml if it exists
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            data = {}
        
        # Update paths - use relative paths from the dataset root
        # The 'path' key should point to the dataset root directory
        # YOLO will append train/val/test to this path
        data['path'] = str(self.dataset_path.resolve())  # Use absolute path
        data['train'] = 'images/train'
        data['val'] = 'images/val'
        data['test'] = 'images/test'
        
        # Preserve or set defaults for nc and names
        if 'nc' not in data:
            data['nc'] = 4  # Default from your config
        if 'names' not in data:
            data['names'] = ['soil', 'bedrock', 'sand', 'big_rock']
        
        # Write updated yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Updated {yaml_path}")
        print(f"\ndata.yaml contents:")
        with open(yaml_path, 'r') as f:
            print(f.read())
    
    def split(self):
        """Execute the full dataset splitting process."""
        print(f"\n{'='*60}")
        print(f"Dataset Splitter")
        print(f"{'='*60}")
        print(f"Original dataset: {self.original_dataset_path}")
        print(f"Split dataset:    {self.dataset_path}")
        print(f"Splits: Train={self.train_split}, Val={self.val_split}, Test={self.test_split}")
        print(f"Random seed: {self.seed}")
        print(f"{'='*60}\n")
        
        # Step 1: Find all image-label pairs in original dataset
        print("Step 1: Finding image-label pairs in original dataset...")
        pairs = self.find_images_and_labels()
        
        # Step 2: Create split dataset directory
        print("\nStep 2: Creating split dataset directory...")
        self.create_split_dataset_directory()
        
        # Step 3: Split the data
        print("\nStep 3: Splitting data...")
        train_pairs, val_pairs, test_pairs = self.split_data(pairs)
        
        # Step 4: Create split directories
        print("\nStep 4: Creating train/val/test directories...")
        self.create_split_directories()
        
        # Step 5: Copy files to split directories
        print("\nStep 5: Copying files to splits...")
        self.copy_files(train_pairs, 'train')
        self.copy_files(val_pairs, 'val')
        self.copy_files(test_pairs, 'test')
        
        # Step 6: Update data.yaml
        print("\nStep 6: Updating data.yaml...")
        self.update_data_yaml()
        
        print(f"\n{'='*60}")
        print("✅ Dataset splitting complete!")
        print(f"{'='*60}")
        print(f"\nOriginal dataset (unchanged):")
        print(f"   📁 {self.original_dataset_path}")
        print(f"\nNew split dataset:")
        print(f"   📁 {self.dataset_path}")
        print(f"\nSplit structure:")
        print(f"   📁 {self.dataset_path / 'images' / 'train'} ({len(train_pairs)} images)")
        print(f"   📁 {self.dataset_path / 'images' / 'val'} ({len(val_pairs)} images)")
        print(f"   📁 {self.dataset_path / 'images' / 'test'} ({len(test_pairs)} images)")
        print(f"\n💡 Update your config to use: {self.dataset_path}")
        print(f"   dataset_root: {self.dataset_path}")



def main():
    parser = argparse.ArgumentParser(
        description='Split YOLO dataset into train/val/test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default 70/20/10 split
  python split_dataset.py /path/to/dataset
  
  # Custom split ratios
  python split_dataset.py /path/to/dataset --train 0.8 --val 0.15 --test 0.05
  
  # With custom seed for reproducibility
  python split_dataset.py /path/to/dataset --seed 123
        """
    )
    
    parser.add_argument('dataset_path', type=str,
                        help='Path to the dataset root directory')
    parser.add_argument('--train', type=float, default=0.7,
                        help='Training split ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--test', type=float, default=0.1,
                        help='Test split ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
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