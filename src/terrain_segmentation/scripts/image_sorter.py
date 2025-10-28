#!/usr/bin/env python3
"""
Image Sorter with Keyboard Shortcuts and YOLO Annotation Overlay
Sorts images into categories: good, questionable, bad, robot_in_frame
Uses keyboard shortcuts: G, Q, B, R, or arrow keys
Shows YOLO segmentation masks overlaid on images
"""

import sys
import shutil
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np


# NAV (Navigation/Traversability) Configuration
NAV_CLASS_INFO = {
    0: {'name': 'soil', 'color': [139, 69, 19]},       # Brown
    1: {'name': 'bedrock', 'color': [128, 128, 128]},  # Gray
    2: {'name': 'sand', 'color': [255, 215, 0]},       # Gold
    3: {'name': 'big_rock', 'color': [255, 105, 180]}, # Pink
}

# GEO (Geology) Configuration
GEO_CLASS_INFO = {
    0: {'name': 'bedrock_massive', 'color': [70, 70, 70]},
    1: {'name': 'bedrock_layered_angled', 'color': [100, 100, 100]},
    2: {'name': 'bedrock_layered_flat', 'color': [130, 130, 130]},
    3: {'name': 'bedrock_layered_unsure', 'color': [160, 160, 160]},
    4: {'name': 'bedrock_conglomerate', 'color': [90, 90, 90]},
    5: {'name': 'bedrock_holey', 'color': [110, 110, 110]},
    6: {'name': 'bedrock_unsure', 'color': [140, 140, 140]},
    10: {'name': 'float_rock_massive', 'color': [255, 105, 180]},
    11: {'name': 'float_rock_layered_angled', 'color': [255, 140, 200]},
    12: {'name': 'float_rock_layered_flat', 'color': [255, 175, 220]},
    13: {'name': 'float_rock_layered_unsure', 'color': [255, 200, 230]},
    14: {'name': 'float_rock_conglomerate', 'color': [255, 120, 190]},
    15: {'name': 'float_rock_holey', 'color': [255, 160, 210]},
    16: {'name': 'float_rock_mixed', 'color': [255, 90, 170]},
    17: {'name': 'float_rock_unsure', 'color': [255, 180, 215]},
    20: {'name': 'sand_dune', 'color': [255, 215, 0]},
    21: {'name': 'sand_ripples', 'color': [255, 235, 100]},
    22: {'name': 'sand_sand', 'color': [255, 245, 150]},
    30: {'name': 'pebbles', 'color': [139, 69, 19]},
    40: {'name': 'vein', 'color': [255, 255, 255]},
    50: {'name': 'hill_peak', 'color': [150, 75, 0]},
}


def detect_config_from_txt(txt_path):
    """Auto-detect whether labels are NAV or GEO by checking class IDs."""
    if not txt_path or not txt_path.exists():
        return 'nav'
    
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(float(parts[0]))
                    if class_id >= 10:  # GEO classes include values >= 10
                        return 'geo'
        return 'nav'
    except:
        return 'nav'


def read_yolo_txt(txt_path, img_width, img_height):
    """Read YOLO segmentation text file and parse polygons."""
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
                    
            except (ValueError, IndexError):
                continue
    
    return polygons


def render_overlay(image, txt_path, alpha=0.5, config='nav'):
    """Render YOLO annotations overlaid on the image."""
    img_width, img_height = image.size
    
    # Read annotations
    polygons = read_yolo_txt(txt_path, img_width, img_height)
    
    if not polygons:
        return image  # No annotations, return original
    
    # Select appropriate class info
    class_info = NAV_CLASS_INFO if config == 'nav' else GEO_CLASS_INFO
    
    # Create mask overlay
    overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw each polygon
    for class_id, points in polygons:
        if class_id in class_info:
            color = class_info[class_id]['color']
            # Add alpha channel
            color_rgba = tuple(color) + (int(alpha * 255),)
            draw.polygon(points, fill=color_rgba, outline=color_rgba)
    
    # Convert original image to RGBA if needed
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Composite the overlay
    result = Image.alpha_composite(image, overlay)
    
    return result.convert('RGB')



class ImageSorter:
    def __init__(self, root, input_folder):
        self.root = root
        self.root.title("Image Sorter with Annotations")
        self.root.geometry("1200x900")
        
        input_path = Path(input_folder)
        self.categories = ['good', 'questionable', 'bad', 'robot_in_frame']
        
        # Determine if user pointed to dataset folder or images folder
        if input_path.name == 'images':
            # User pointed directly to images folder
            self.input_folder = input_path
            self.labels_folder = input_path.parent / 'labels'
        elif (input_path / 'images').exists():
            # User pointed to dataset folder containing images/ and labels/
            self.input_folder = input_path / 'images'
            self.labels_folder = input_path / 'labels'
        else:
            # Assume it's just a regular folder with images
            self.input_folder = input_path
            self.labels_folder = input_path / 'labels'
        
        print(f"Images folder: {self.input_folder}")
        
        if not self.labels_folder.exists():
            print(f"Warning: Labels folder not found at {self.labels_folder}")
            print("Will display images without annotations.")
            self.labels_folder = None
        else:
            print(f"Found labels folder: {self.labels_folder}")
        
        # Auto-detect config type
        self.config = 'nav'
        if self.labels_folder:
            txt_files = list(self.labels_folder.glob('*.txt'))
            if txt_files:
                self.config = detect_config_from_txt(txt_files[0])
                print(f"Detected config: {self.config.upper()}")
        
        # Overlay transparency
        self.alpha = 0.5
        
        # Create category folders
        for category in self.categories:
            (self.input_folder / category).mkdir(exist_ok=True)
        
        # Get list of unsorted images (not in any category folder)
        self.images = self.get_unsorted_images()
        self.current_index = 0
        
        if not self.images:
            messagebox.showinfo("Done", "No unsorted images found!")
            root.quit()
            return
        
        # Setup UI
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.root.bind('g', lambda e: self.sort_image('good'))
        self.root.bind('G', lambda e: self.sort_image('good'))
        self.root.bind('q', lambda e: self.sort_image('questionable'))
        self.root.bind('Q', lambda e: self.sort_image('questionable'))
        self.root.bind('b', lambda e: self.sort_image('bad'))
        self.root.bind('B', lambda e: self.sort_image('bad'))
        self.root.bind('r', lambda e: self.sort_image('robot_in_frame'))
        self.root.bind('R', lambda e: self.sort_image('robot_in_frame'))
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Escape>', lambda e: root.quit())
        self.root.bind('+', lambda e: self.change_alpha(0.1))
        self.root.bind('=', lambda e: self.change_alpha(0.1))
        self.root.bind('-', lambda e: self.change_alpha(-0.1))
        self.root.bind('_', lambda e: self.change_alpha(-0.1))
        
        # Load first image
        self.display_image()
    
    def get_unsorted_images(self):
        """Get list of images that haven't been sorted yet."""
        all_images = []
        
        # Get all image files in root folder (not in subfolders)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            all_images.extend(self.input_folder.glob(ext))
        
        # Filter out images in category folders
        unsorted = []
        for img in all_images:
            # Check if image is in root folder (not in a category subfolder)
            if img.parent == self.input_folder:
                unsorted.append(img)
        
        return sorted(unsorted)
    
    def setup_ui(self):
        """Create the user interface."""
        # Title and progress
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=10)
        
        self.progress_label = tk.Label(
            self.info_frame, 
            text="",
            font=('Arial', 14, 'bold')
        )
        self.progress_label.pack()
        
        self.filename_label = tk.Label(
            self.info_frame,
            text="",
            font=('Arial', 10)
        )
        self.filename_label.pack()
        
        # Image display
        self.image_label = tk.Label(self.root, bg='gray')
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Category buttons with color coding
        btn_configs = [
            ('Good [G]', 'good', '#4CAF50'),
            ('Questionable [Q]', 'questionable', '#FFC107'),
            ('Bad [B]', 'bad', '#F44336'),
            ('Robot in Frame [R]', 'robot_in_frame', '#2196F3')
        ]
        
        for text, category, color in btn_configs:
            btn = tk.Button(
                button_frame,
                text=text,
                command=lambda c=category: self.sort_image(c),
                bg=color,
                fg='white',
                font=('Arial', 12, 'bold'),
                width=18,
                height=2
            )
            btn.pack(side=tk.LEFT, padx=5)
        
        # Navigation frame
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=5)
        
        tk.Button(
            nav_frame,
            text="◄ Previous [←]",
            command=self.previous_image,
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            nav_frame,
            text="Next [→]",
            command=self.next_image,
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=5)
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="Keyboard: G=Good | Q=Questionable | B=Bad | R=Robot | ←/→=Navigate | +/-=Overlay | ESC=Quit",
            font=('Arial', 9),
            fg='gray'
        )
        instructions.pack(pady=5)
    
    def display_image(self):
        """Display the current image with annotations overlay."""
        if not self.images or self.current_index >= len(self.images):
            messagebox.showinfo("Complete", "All images have been sorted!")
            self.root.quit()
            return
        
        # Update progress
        self.progress_label.config(
            text=f"Image {self.current_index + 1} of {len(self.images)} | Overlay: {int(self.alpha*100)}%"
        )
        self.filename_label.config(
            text=self.images[self.current_index].name
        )
        
        # Load and display image
        try:
            img = Image.open(self.images[self.current_index])
            
            # Find corresponding label file
            if self.labels_folder:
                base_name = self.images[self.current_index].stem
                txt_path = self.labels_folder / f"{base_name}.txt"
                
                if txt_path.exists():
                    # Render with annotations overlay
                    img = render_overlay(img, txt_path, alpha=self.alpha, config=self.config)
                else:
                    print(f"Warning: No label file found for {base_name}")
            
            # Resize to fit window while maintaining aspect ratio
            display_width = 1180
            display_height = 650
            
            img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            self.next_image()
    
    def change_alpha(self, delta):
        """Change overlay transparency."""
        self.alpha = max(0.0, min(1.0, self.alpha + delta))
        print(f"Overlay transparency: {int(self.alpha * 100)}%")
        self.display_image()
    
    def sort_image(self, category):
        """Move current image and its label to the specified category folder."""
        if self.current_index >= len(self.images):
            return
        
        current_image = self.images[self.current_index]
        destination = self.input_folder / category / current_image.name
        
        try:
            # Move the image file
            shutil.move(str(current_image), str(destination))
            print(f"Moved {current_image.name} to {category}/")
            
            # Also move the corresponding label file if it exists
            if self.labels_folder:
                base_name = current_image.stem
                txt_path = self.labels_folder / f"{base_name}.txt"
                
                if txt_path.exists():
                    # Create labels subfolder in category if needed
                    label_category_dir = self.input_folder / category / 'labels'
                    label_category_dir.mkdir(exist_ok=True)
                    
                    label_destination = label_category_dir / txt_path.name
                    shutil.move(str(txt_path), str(label_destination))
                    print(f"Moved {txt_path.name} to {category}/labels/")
            
            # Remove from list and show next image
            self.images.pop(self.current_index)
            
            # Adjust index if needed
            if self.current_index >= len(self.images) and self.current_index > 0:
                self.current_index -= 1
            
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to move image: {e}")
    
    def next_image(self):
        """Go to next image without sorting current one."""
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.display_image()
    
    def previous_image(self):
        """Go to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()


def main():
    if len(sys.argv) < 2:
        print("Usage: python image_sorter.py /path/to/dataset")
        print("   or: python image_sorter.py /path/to/dataset/images")
        print("\nThis script works with YOLO format datasets:")
        print("  - Point it to the dataset folder (containing images/ and labels/)")
        print("  - OR point it directly to the images/ folder")
        print("  - Annotations will be overlaid on images as colored masks")
        print("\nThis will create 4 subfolders inside images/:")
        print("  good, questionable, bad, robot_in_frame")
        print("Images AND their labels will be MOVED (not copied) into these folders.")
        print("\nKeyboard shortcuts:")
        print("  G - Good")
        print("  Q - Questionable")
        print("  B - Bad")
        print("  R - Robot in frame")
        print("  ← → - Navigate between images")
        print("  + - - Adjust overlay transparency")
        print("  ESC - Quit")
        sys.exit(1)
    
    input_folder = Path(sys.argv[1])
    
    if not input_folder.exists():
        print(f"Error: Folder not found: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: Not a directory: {input_folder}")
        sys.exit(1)
    
    root = tk.Tk()
    app = ImageSorter(root, input_folder)
    root.mainloop()


if __name__ == '__main__':
    main()