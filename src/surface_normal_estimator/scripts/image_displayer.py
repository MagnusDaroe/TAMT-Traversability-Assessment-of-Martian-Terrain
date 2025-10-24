#!/usr/bin/env python3

import matplotlib.pyplot as plt
import cv2
import os
import sys

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (surface_normal_estimator)
    parent_dir = os.path.dirname(script_dir)
    
    # Path to the saved image
    image_path = os.path.join(parent_dir, 'images', 'surface_normals_viz.png')
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Please run the image_saver node first to generate the image.")
        sys.exit(1)
    
    # Load the image
    # OpenCV loads as BGR, convert to RGB for matplotlib
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.figure(figsize=(12, 9))
    plt.imshow(image_rgb)
    plt.title('Surface Normals Visualization', fontsize=16)
    plt.axis('off')
    
    # Add colorbar explanation
    plt.figtext(0.5, 0.02, 
                'R = X-component (left-right) | G = Y-component (up-down) | B = Z-component (forward-backward)',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed image from: {image_path}")
    print(f"Image shape: {image_rgb.shape}")
    print(f"Image dtype: {image_rgb.dtype}")

if __name__ == '__main__':
    main()