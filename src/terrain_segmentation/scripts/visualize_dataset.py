#!/usr/bin/env python3
"""
Interactive viewer for YOLO semantic segmentation labels.
Shows images in a window with keyboard navigation.
Usage: python view_yolo_labels.py /path/to/yolo_format
"""

import sys
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Class definitions
CLASS_INFO = {
    0: {'name': 'soil', 'color': [139, 69, 19]},       # Brown
    1: {'name': 'bedrock', 'color': [128, 128, 128]},  # Gray
    2: {'name': 'sand', 'color': [255, 215, 0]},       # Gold
    3: {'name': 'big_rock', 'color': [255, 105, 180]},    # Pink
    255: {'name': 'null', 'color': [0, 0, 0]}          # Black
}


def label_to_color(label_array):
    """Convert grayscale label to RGB colored visualization."""
    h, w = label_array.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, info in CLASS_INFO.items():
        mask = label_array == class_id
        color_img[mask] = info['color']
    
    return color_img


class InteractiveViewer:
    """Interactive viewer for navigating through dataset."""
    
    def __init__(self, yolo_format_dir, alpha=0.5):
        self.yolo_format_dir = Path(yolo_format_dir)
        self.images_dir = self.yolo_format_dir / 'images'
        self.labels_dir = self.yolo_format_dir / 'labels'
        self.alpha = alpha
        
        # Get all label files
        self.label_files = sorted(self.labels_dir.glob('*.png'))
        self.current_idx = 0
        
        if len(self.label_files) == 0:
            print(f"No label files found in {self.labels_dir}")
            sys.exit(1)
        
        print(f"Found {len(self.label_files)} samples")
        print("\nControls:")
        print("  → or Space : Next image")
        print("  ← : Previous image")
        print("  Q : Quit")
        print("  + : Increase overlay opacity")
        print("  - : Decrease overlay opacity")
        print()
        
        # Create figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Display first image
        self.display_current()
        plt.show()
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'right' or event.key == ' ':
            self.current_idx = (self.current_idx + 1) % len(self.label_files)
            self.display_current()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.label_files)
            self.display_current()
        elif event.key == '+' or event.key == '=':
            self.alpha = min(1.0, self.alpha + 0.1)
            print(f"Overlay opacity: {self.alpha:.1f}")
            self.display_current()
        elif event.key == '-' or event.key == '_':
            self.alpha = max(0.0, self.alpha - 0.1)
            print(f"Overlay opacity: {self.alpha:.1f}")
            self.display_current()
        elif event.key == 'q':
            plt.close(self.fig)
    
    def display_current(self):
        """Display the current image."""
        label_path = self.label_files[self.current_idx]
        base_name = label_path.stem
        
        # Find matching image
        img_path = None
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            potential_img = self.images_dir / f"{base_name}{ext}"
            if potential_img.exists():
                img_path = potential_img
                break
        
        if img_path is None:
            print(f"Warning: No matching image for {base_name}")
            return
        
        try:
            # Load image and label
            image = np.array(Image.open(img_path))
            label = np.array(Image.open(label_path))
            
            # Convert label to color
            label_color = label_to_color(label)
            
            # Clear previous plots
            for ax in self.axes:
                ax.clear()
            
            # Original image
            self.axes[0].imshow(image)
            self.axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            self.axes[0].axis('off')
            
            # Label only
            self.axes[1].imshow(label_color)
            self.axes[1].set_title('Segmentation Labels', fontsize=14, fontweight='bold')
            self.axes[1].axis('off')
            
            # Overlay
            if len(image.shape) == 2:  # Grayscale image
                image = np.stack([image, image, image], axis=-1)
            overlay = (image.astype(float) * (1 - self.alpha) + 
                      label_color.astype(float) * self.alpha).astype(np.uint8)
            self.axes[2].imshow(overlay)
            self.axes[2].set_title(f'Overlay (α={self.alpha:.1f})', fontsize=14, fontweight='bold')
            self.axes[2].axis('off')
            
            # Update figure title
            self.fig.suptitle(f'Sample {self.current_idx + 1}/{len(self.label_files)}: {base_name}', 
                            fontsize=16, fontweight='bold')
            
            # Create legend
            legend_elements = []
            for class_id in sorted([k for k in CLASS_INFO.keys() if k != 255]):
                info = CLASS_INFO[class_id]
                color_normalized = [c/255.0 for c in info['color']]
                legend_elements.append(
                    mpatches.Patch(color=color_normalized, 
                                 label=info['name'].replace('_', ' ').title())
                )
            
            # Remove old legend if exists
            if hasattr(self, 'legend'):
                self.legend.remove()
            
            self.legend = self.fig.legend(handles=legend_elements, loc='lower center', 
                                        ncol=4, fontsize=12, frameon=True, fancybox=True)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1, top=0.93)
            self.fig.canvas.draw()
            
        except Exception as e:
            print(f"Error displaying {base_name}: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive viewer for YOLO semantic segmentation labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  → or Space : Next image
  ← : Previous image
  + : Increase overlay opacity
  - : Decrease overlay opacity
  Q : Quit

Example:
  %(prog)s /path/to/yolo_format
  %(prog)s /path/to/yolo_format --alpha 0.7
        """
    )
    
    parser.add_argument(
        'yolo_dir',
        type=str,
        help='Path to yolo_format directory (contains images/ and labels/)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Initial overlay transparency 0-1 (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    yolo_dir = Path(args.yolo_dir)
    
    if not yolo_dir.exists():
        print(f"Error: Directory not found: {yolo_dir}")
        sys.exit(1)
    
    images_dir = yolo_dir / 'images'
    labels_dir = yolo_dir / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: Expected structure not found in {yolo_dir}")
        print(f"  images/ exists: {images_dir.exists()}")
        print(f"  labels/ exists: {labels_dir.exists()}")
        sys.exit(1)
    
    # Start interactive viewer
    viewer = InteractiveViewer(yolo_dir, args.alpha)


if __name__ == '__main__':
    main()