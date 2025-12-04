#!/usr/bin/env python3
"""
Image Folder Color Analysis for Isaac Sim
Analyzes a folder of images and outputs color statistics and augmentation parameters
that can be used in Isaac Sim for material/asset color modification.
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple


def rgb_to_lab(image):
    """Convert RGB image to LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab.astype(np.float32)


def lab_to_rgb(lab_color):
    """Convert LAB color to RGB."""
    # Create a 1x1 image with the LAB color
    lab_img = np.uint8([[lab_color]])
    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    return rgb_img[0][0]


def get_color_statistics(image):
    """
    Calculate mean and standard deviation for each channel in LAB space.
    Returns statistics for L, A, and B channels.
    """
    l_channel, a_channel, b_channel = cv2.split(image)
    
    stats = {
        'l_mean': float(np.mean(l_channel)),
        'l_std': float(np.std(l_channel)),
        'a_mean': float(np.mean(a_channel)),
        'a_std': float(np.std(a_channel)),
        'b_mean': float(np.mean(b_channel)),
        'b_std': float(np.std(b_channel))
    }
    
    return stats


def get_dominant_colors_kmeans(image, n_colors=5):
    """
    Extract dominant colors using K-means clustering.
    Returns colors in RGB format (0-255 range).
    """
    # Reshape image to be a list of pixels
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # Apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, 
                                     cv2.KMEANS_PP_CENTERS)
    
    # Convert to uint8 and BGR to RGB
    centers = centers.astype(np.uint8)
    
    # Count pixels in each cluster to get color frequencies
    unique, counts = np.unique(labels, return_counts=True)
    frequencies = counts / len(labels)
    
    # Sort by frequency
    sorted_indices = np.argsort(-frequencies)
    dominant_colors = centers[sorted_indices]
    color_frequencies = frequencies[sorted_indices]
    
    return dominant_colors, color_frequencies


def bgr_to_rgb_normalized(bgr_color):
    """Convert BGR color to normalized RGB (0-1 range) for Isaac Sim."""
    return [float(bgr_color[2]/255.0), float(bgr_color[1]/255.0), float(bgr_color[0]/255.0)]


def analyze_image_folder(folder_path: Path, n_dominant_colors: int = 5) -> Dict:
    """
    Analyze all images in a folder and compute aggregate statistics.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in folder_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No images found in {folder_path}")
    
    print(f"\nFound {len(image_files)} images in {folder_path}")
    print("="*60)
    
    all_stats = []
    all_dominant_colors = []
    all_color_weights = []
    
    for img_path in image_files:
        print(f"Analyzing: {img_path.name}")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Could not load {img_path.name}, skipping...")
            continue
        
        # Get LAB statistics
        lab = rgb_to_lab(img)
        stats = get_color_statistics(lab)
        all_stats.append(stats)
        
        # Get dominant colors
        dominant_colors, frequencies = get_dominant_colors_kmeans(img, n_dominant_colors)
        all_dominant_colors.append(dominant_colors)
        all_color_weights.append(frequencies)
    
    if not all_stats:
        raise ValueError("No images could be loaded successfully")
    
    # Compute aggregate statistics
    aggregate_stats = {
        'l_mean': np.mean([s['l_mean'] for s in all_stats]),
        'l_std': np.mean([s['l_std'] for s in all_stats]),
        'a_mean': np.mean([s['a_mean'] for s in all_stats]),
        'a_std': np.mean([s['a_std'] for s in all_stats]),
        'b_mean': np.mean([s['b_mean'] for s in all_stats]),
        'b_std': np.mean([s['b_std'] for s in all_stats]),
    }
    
    # Compute overall dominant colors (weighted average across all images)
    all_colors_flat = np.vstack(all_dominant_colors)
    all_weights_flat = np.hstack(all_color_weights) / len(all_color_weights)
    
    # Cluster again to get final dominant colors
    final_dominant, final_frequencies = get_dominant_colors_kmeans(
        all_colors_flat, n_dominant_colors
    )
    
    return {
        'num_images': len(all_stats),
        'aggregate_stats': aggregate_stats,
        'dominant_colors_bgr': final_dominant.tolist(),
        'dominant_colors_rgb_normalized': [bgr_to_rgb_normalized(c) for c in final_dominant],
        'color_frequencies': final_frequencies.tolist(),
        'individual_stats': all_stats
    }


def create_isaac_sim_config(analysis_results: Dict) -> Dict:
    """
    Create Isaac Sim compatible color augmentation configuration.
    """
    stats = analysis_results['aggregate_stats']
    
    # Convert LAB mean to RGB for base color
    lab_mean = np.array([stats['l_mean'], stats['a_mean'], stats['b_mean']])
    base_color_bgr = lab_to_rgb(lab_mean)
    base_color_rgb_normalized = bgr_to_rgb_normalized(base_color_bgr)
    
    config = {
        'isaac_sim_color_augmentation': {
            'base_color_rgb': base_color_rgb_normalized,
            'base_color_rgb_255': [int(c*255) for c in base_color_rgb_normalized],
            
            'dominant_colors': [
                {
                    'rgb_normalized': color,
                    'rgb_255': [int(c*255) for c in color],
                    'frequency': float(freq)
                }
                for color, freq in zip(
                    analysis_results['dominant_colors_rgb_normalized'],
                    analysis_results['color_frequencies']
                )
            ],
            
            'color_variation_params': {
                'hue_shift_range': [-0.1, 0.1],  # Adjust based on your needs
                'saturation_scale_range': [0.8, 1.2],
                'brightness_scale_range': [
                    max(0.5, 1.0 - stats['l_std']/255.0),
                    min(1.5, 1.0 + stats['l_std']/255.0)
                ],
            },
            
            'lab_statistics': {
                'lightness': {'mean': stats['l_mean'], 'std': stats['l_std']},
                'a_channel': {'mean': stats['a_mean'], 'std': stats['a_std']},
                'b_channel': {'mean': stats['b_mean'], 'std': stats['b_std']},
            }
        }
    }
    
    return config


def print_analysis_summary(analysis_results: Dict):
    """Print a human-readable summary of the analysis."""
    stats = analysis_results['aggregate_stats']
    
    print("\n" + "="*60)
    print(f"ANALYSIS SUMMARY ({analysis_results['num_images']} images)")
    print("="*60)
    
    print(f"\nAGGREGATE COLOR STATISTICS (LAB Color Space):")
    print(f"  Lightness:       Mean={stats['l_mean']:.2f}, StdDev={stats['l_std']:.2f}")
    print(f"  A (Green-Red):   Mean={stats['a_mean']:.2f}, StdDev={stats['a_std']:.2f}")
    print(f"  B (Blue-Yellow): Mean={stats['b_mean']:.2f}, StdDev={stats['b_std']:.2f}")
    
    print(f"\nCOLOR INTERPRETATION:")
    if stats['a_mean'] > 128:
        print(f"  - Tends toward RED tones")
    else:
        print(f"  - Tends toward GREEN tones")
    
    if stats['b_mean'] > 128:
        print(f"  - Tends toward YELLOW tones")
    else:
        print(f"  - Tends toward BLUE tones")
    
    if stats['l_mean'] > 170:
        print(f"  - Very BRIGHT images")
    elif stats['l_mean'] > 128:
        print(f"  - Moderately bright images")
    elif stats['l_mean'] > 85:
        print(f"  - Moderately dark images")
    else:
        print(f"  - Very DARK images")
    
    print(f"\nDOMINANT COLORS (RGB, 0-255 range):")
    for i, (color, freq) in enumerate(zip(
        analysis_results['dominant_colors_rgb_normalized'],
        analysis_results['color_frequencies']
    ), 1):
        rgb_255 = [int(c*255) for c in color]
        print(f"  {i}. RGB({rgb_255[0]:3d}, {rgb_255[1]:3d}, {rgb_255[2]:3d}) - {freq*100:.1f}% of pixels")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze image folder colors for Isaac Sim augmentation'
    )
    parser.add_argument('--folder', '-f', required=True,
                       help='Path to folder containing images')
    parser.add_argument('--output', '-o', default='isaac_sim_colors.json',
                       help='Output JSON file (default: isaac_sim_colors.json)')
    parser.add_argument('--n-colors', '-n', type=int, default=5,
                       help='Number of dominant colors to extract (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed per-image statistics')
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    if not folder_path.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        return
    
    # Analyze the folder
    print(f"Analyzing images in: {folder_path}")
    analysis_results = analyze_image_folder(folder_path, args.n_colors)
    
    # Create Isaac Sim config
    isaac_config = create_isaac_sim_config(analysis_results)
    
    # Print summary
    print_analysis_summary(analysis_results)
    
    # Save results
    output_data = {
        'analysis_results': analysis_results,
        'isaac_sim_config': isaac_config
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    print(f"\nTo use in Isaac Sim:")
    print(f"  1. Load the JSON file")
    print(f"  2. Use 'dominant_colors' for material color randomization")
    print(f"  3. Use 'color_variation_params' for augmentation ranges")
    print(f"  4. RGB values are normalized to [0,1] range (Isaac Sim format)")


if __name__ == "__main__":
    main()