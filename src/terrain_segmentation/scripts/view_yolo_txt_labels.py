#!/usr/bin/env python3
"""
Interactive viewer for YOLO semantic segmentation labels (text format).
Shows images in a window with keyboard navigation.
Reads .txt files with polygon coordinates and renders them as segmentation masks.
Supports both NAV and GEO configurations from AI4Mars dataset.

Usage: python view_yolo_txt_labels.py /path/to/yolo_format [--config nav|geo]
"""

import sys
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# NAV (Navigation/Traversability) Configuration
# =============================================================================
NAV_CLASS_INFO = {
    0: {'name': 'soil', 'color': [139, 69, 19]},       # Brown
    1: {'name': 'bedrock', 'color': [128, 128, 128]},  # Gray
    2: {'name': 'sand', 'color': [255, 215, 0]},       # Gold
    3: {'name': 'big_rock', 'color': [255, 105, 180]}, # Pink
}

# =============================================================================
# GEO (Geology) Configuration
# =============================================================================
GEO_CLASS_INFO = {
    # Bedrock types (0-6)
    0: {'name': 'bedrock_massive', 'color': [70, 70, 70]},
    1: {'name': 'bedrock_layered_angled', 'color': [100, 100, 100]},
    2: {'name': 'bedrock_layered_flat', 'color': [130, 130, 130]},
    3: {'name': 'bedrock_layered_unsure', 'color': [160, 160, 160]},
    4: {'name': 'bedrock_conglomerate', 'color': [90, 90, 90]},
    5: {'name': 'bedrock_holey', 'color': [110, 110, 110]},
    6: {'name': 'bedrock_unsure', 'color': [140, 140, 140]},
    
    # Float rocks (10-17)
    10: {'name': 'float_rock_massive', 'color': [255, 105, 180]},        # Pink
    11: {'name': 'float_rock_layered_angled', 'color': [255, 140, 200]},
    12: {'name': 'float_rock_layered_flat', 'color': [255, 175, 220]},
    13: {'name': 'float_rock_layered_unsure', 'color': [255, 200, 230]},
    14: {'name': 'float_rock_conglomerate', 'color': [255, 120, 190]},
    15: {'name': 'float_rock_holey', 'color': [255, 160, 210]},
    16: {'name': 'float_rock_mixed', 'color': [255, 90, 170]},
    17: {'name': 'float_rock_unsure', 'color': [255, 180, 215]},
    
    # Sand types (20-22)
    20: {'name': 'sand_dune', 'color': [255, 215, 0]},           # Gold
    21: {'name': 'sand_ripples', 'color': [255, 235, 100]},      # Light gold
    22: {'name': 'sand_sand', 'color': [255, 245, 150]},         # Pale gold
    
    # Other geological features (30-50)
    30: {'name': 'pebbles', 'color': [139, 69, 19]},             # Brown
    40: {'name': 'vein', 'color': [255, 255, 255]},              # White
    50: {'name': 'hill_peak', 'color': [150, 75, 0]},            # Dark brown
}


def detect_config_from_classes(label_path):
    """
    Auto-detect whether this is a NAV or GEO dataset by checking class IDs in the file.
    
    Args:
        label_path: Path to a .txt label file
        
    Returns:
        'nav' or 'geo'
    """
    if not label_path.exists():
        return 'nav'  # Default
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(float(parts[0]))
                    # GEO classes include values >= 10
                    if class_id >= 10:
                        return 'geo'
        return 'nav'
    except:
        return 'nav'


def detect_config_from_dir(yolo_dir):
    """
    Try to detect configuration from directory name.
    
    Args:
        yolo_dir: Path to YOLO format directory
        
    Returns:
        'nav', 'geo', or None (if can't determine)
    """
    dir_name = yolo_dir.name.lower()
    if 'nav' in dir_name:
        return 'nav'
    elif 'geo' in dir_name:
        return 'geo'
    return None


def read_yolo_txt(txt_path, img_width, img_height):
    """
    Read YOLO segmentation text file and parse polygons.
    
    Args:
        txt_path: Path to .txt file
        img_width: Image width for denormalization
        img_height: Image height for denormalization
        
    Returns:
        List of (class_id, polygon_points) tuples
    """
    polygons = []
    
    if not txt_path.exists():
        return polygons
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 7:  # Need at least class_id + 3 points (6 coords)
                continue
            
            try:
                class_id = int(float(parts[0]))
                
                # Parse normalized coordinates
                coords = [float(x) for x in parts[1:]]
                
                # Group into (x, y) pairs and denormalize
                points = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        x = coords[i] * img_width
                        y = coords[i + 1] * img_height
                        points.append((x, y))
                
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    polygons.append((class_id, points))
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse line in {txt_path.name}: {line[:50]}...")
                continue
    
    return polygons


def render_polygons_to_mask(polygons, img_width, img_height):
    """
    Render polygons to a segmentation mask.
    
    Args:
        polygons: List of (class_id, points) tuples
        img_width: Image width
        img_height: Image height
        
    Returns:
        Numpy array of shape (height, width) with class IDs
    """
    # Create a PIL image for drawing (faster than numpy for polygons)
    mask_img = Image.new('L', (img_width, img_height), 255)  # Start with 255 (background)
    draw = ImageDraw.Draw(mask_img)
    
    # Sort polygons by class ID to ensure consistent ordering
    # Draw in order: higher class IDs might overlap lower ones
    polygons = sorted(polygons, key=lambda x: x[0])
    
    for class_id, points in polygons:
        # Draw filled polygon (will use appropriate class_info in rendering)
        draw.polygon(points, fill=class_id, outline=class_id)
    
    # Convert to numpy array
    mask = np.array(mask_img)
    return mask


def label_to_color(label_array, class_info):
    """Convert grayscale label to RGB colored visualization."""
    h, w = label_array.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Background (255 or any undefined class) remains black
    for class_id, info in class_info.items():
        mask = label_array == class_id
        color_img[mask] = info['color']
    
    return color_img


class InteractiveViewer:
    """Interactive viewer for navigating through dataset."""
    
    def __init__(self, yolo_format_dir, alpha=0.5, config=None):
        self.yolo_format_dir = Path(yolo_format_dir)
        self.images_dir = self.yolo_format_dir / 'images'
        self.labels_dir = self.yolo_format_dir / 'labels'
        self.alpha = alpha
        
        # Try to detect config
        if config is None:
            config = detect_config_from_dir(self.yolo_format_dir)
            if config is None:
                # Try to detect from first label file
                label_files = list(self.labels_dir.glob('*.txt'))
                if len(label_files) > 0:
                    config = detect_config_from_classes(label_files[0])
                else:
                    config = 'nav'  # Default
            print(f"Auto-detected configuration: {config.upper()}")
        
        self.config = config
        self.class_info = NAV_CLASS_INFO if config == 'nav' else GEO_CLASS_INFO
        
        # Get all txt label files
        self.label_files = sorted(self.labels_dir.glob('*.txt'))
        self.current_idx = 0
        
        if len(self.label_files) == 0:
            print(f"No .txt label files found in {self.labels_dir}")
            sys.exit(1)
        
        print(f"Found {len(self.label_files)} samples with .txt labels")
        print(f"Using {config.upper()} configuration ({len(self.class_info)} classes)")
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
        
        # Also check for corresponding PNG label to get dimensions
        label_png_path = self.labels_dir / f"{base_name}.png"
        
        if img_path is None:
            print(f"Warning: No matching image for {base_name}")
            return
        
        try:
            # Load image
            image = np.array(Image.open(img_path))
            
            # Get dimensions from the label PNG if it exists, otherwise use image dimensions
            if label_png_path.exists():
                label_png = Image.open(label_png_path)
                label_width, label_height = label_png.size
            else:
                label_height, label_width = image.shape[:2]
            
            # Read YOLO txt file and parse polygons (using label dimensions!)
            polygons = read_yolo_txt(label_path, label_width, label_height)
            
            if len(polygons) == 0:
                print(f"Warning: No valid polygons found in {label_path.name}")
            
            # Render polygons to mask (using label dimensions!)
            label = render_polygons_to_mask(polygons, label_width, label_height)
            
            # Convert label to color using appropriate class info
            label_color = label_to_color(label, self.class_info)
            
            # Clear previous plots
            for ax in self.axes:
                ax.clear()
            
            # Resize image to match label dimensions if needed
            if image.shape[:2] != (label_height, label_width):
                from PIL import Image as PILImage
                if len(image.shape) == 2:
                    image_pil = PILImage.fromarray(image, mode='L')
                else:
                    image_pil = PILImage.fromarray(image)
                image_pil = image_pil.resize((label_width, label_height), PILImage.LANCZOS)
                image = np.array(image_pil)
            
            # Original image
            self.axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
            self.axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            self.axes[0].axis('off')
            
            # Label only
            self.axes[1].imshow(label_color)
            self.axes[1].set_title(f'Segmentation from TXT ({len(polygons)} polygons)', 
                                  fontsize=14, fontweight='bold')
            self.axes[1].axis('off')
            
            # Overlay
            if len(image.shape) == 2:  # Grayscale image
                image_rgb = np.stack([image, image, image], axis=-1)
            else:
                image_rgb = image
            
            overlay = (image_rgb.astype(float) * (1 - self.alpha) + 
                      label_color.astype(float) * self.alpha).astype(np.uint8)
            self.axes[2].imshow(overlay)
            self.axes[2].set_title(f'Overlay (α={self.alpha:.1f})', fontsize=14, fontweight='bold')
            self.axes[2].axis('off')
            
            # Update figure title
            config_name = "Navigation" if self.config == 'nav' else "Geology"
            self.fig.suptitle(f'{config_name} | Sample {self.current_idx + 1}/{len(self.label_files)}: {base_name}', 
                            fontsize=16, fontweight='bold')
            
            # Create legend with class counts
            class_counts = {}
            for class_id, _ in polygons:
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            legend_elements = []
            for class_id in sorted(self.class_info.keys()):
                info = self.class_info[class_id]
                count = class_counts.get(class_id, 0)
                if count > 0:  # Only show classes that are present
                    color_normalized = [c/255.0 for c in info['color']]
                    label_text = f"{info['name'].replace('_', ' ').title()} ({count})"
                    legend_elements.append(
                        mpatches.Patch(color=color_normalized, label=label_text)
                    )
            
            # Remove old legend if exists
            if hasattr(self, 'legend') and self.legend:
                self.legend.remove()
            
            if legend_elements:
                # Adjust number of columns based on number of classes
                ncols = min(4, len(legend_elements))
                self.legend = self.fig.legend(handles=legend_elements, loc='lower center', 
                                            ncol=ncols, fontsize=10, frameon=True, 
                                            fancybox=True, shadow=True)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12, top=0.93)
            self.fig.canvas.draw()
            
        except Exception as e:
            print(f"Error displaying {base_name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive viewer for YOLO semantic segmentation labels (TXT format)\n'
                    'Supports both NAV and GEO configurations from AI4Mars dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  → or Space : Next image
  ← : Previous image
  + : Increase overlay opacity
  - : Decrease overlay opacity
  Q : Quit

Examples:
  %(prog)s /path/to/ai4mars_nav
  %(prog)s /path/to/ai4mars_geo --alpha 0.7
  %(prog)s /path/to/yolo_format --config nav
        """
    )
    
    parser.add_argument(
        'yolo_dir',
        type=str,
        help='Path to yolo_format directory (contains images/ and labels/ with .txt files)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Initial overlay transparency 0-1 (default: 0.5)'
    )
    parser.add_argument(
        '--config',
        type=str,
        choices=['nav', 'geo'],
        default=None,
        help='Force configuration type (auto-detected if not specified)'
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
    viewer = InteractiveViewer(yolo_dir, args.alpha, args.config)


if __name__ == '__main__':
    main()