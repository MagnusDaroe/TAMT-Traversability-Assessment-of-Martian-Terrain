#!/usr/bin/env python3
"""
YOLO Model Inspector
Shows model architecture and identifies class-specific layers
"""

import torch
from ultralytics import YOLO
import argparse
from pathlib import Path


def inspect_yolo_model(model_path):
    """Inspect YOLO model architecture and weights"""
    
    print(f"\n{'='*80}")
    print("YOLO Model Inspector")
    print(f"{'='*80}\n")
    
    model_path = Path(model_path).expanduser()
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    print(f"Loading: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"✓ Model loaded\n")
    
    # Basic info
    print(f"{'='*80}")
    print("Model Information")
    print(f"{'='*80}")
    print(f"Number of classes: {model.model.nc}")
    print(f"Class names: {model.model.names}")
    print(f"Task: {model.task}")
    print()
    
    # Architecture summary
    print(f"{'='*80}")
    print("Architecture Summary")
    print(f"{'='*80}")
    
    for i, module in enumerate(model.model.model):
        module_type = type(module).__name__
        
        # Identify which section
        if i < 10:
            section = "BACKBONE"
        elif i < 23:
            section = "NECK"
        else:
            section = "HEAD (CLASS-SPECIFIC)"
        
        # Count parameters
        params = sum(p.numel() for p in module.parameters())
        
        print(f"Layer {i:2d} [{section:20s}] {module_type:30s} {params:>10,} params")
    
    print()
    
    # State dict keys (weight names)
    print(f"{'='*80}")
    print("State Dict Keys (Weight Names)")
    print(f"{'='*80}")
    
    state_dict = model.model.state_dict()
    
    # Group by layer
    layer_groups = {}
    for key in state_dict.keys():
        # Extract layer number (e.g., "model.23.cv2" -> 23)
        parts = key.split('.')
        if len(parts) >= 2 and parts[0] == 'model' and parts[1].isdigit():
            layer_num = int(parts[1])
            if layer_num not in layer_groups:
                layer_groups[layer_num] = []
            layer_groups[layer_num].append(key)
    
    # Show layers with their keys
    for layer_num in sorted(layer_groups.keys()):
        keys = layer_groups[layer_num]
        
        # Determine if class-specific
        is_head = layer_num >= 23
        marker = "🔴 CLASS-SPECIFIC" if is_head else "✓ Class-agnostic"
        
        print(f"\nLayer {layer_num} {marker}")
        print(f"  Keys ({len(keys)}):")
        
        # Show first 5 keys
        for key in keys[:5]:
            shape = state_dict[key].shape
            print(f"    {key:60s} {str(shape):20s}")
        
        if len(keys) > 5:
            print(f"    ... and {len(keys)-5} more")
    
    print()
    
    # Identify class-specific patterns
    print(f"{'='*80}")
    print("Class-Specific Layer Patterns")
    print(f"{'='*80}\n")
    
    class_specific_keys = [k for k in state_dict.keys() if 'model.23' in k or 'model.22' in k]
    
    print(f"Found {len(class_specific_keys)} class-specific weight tensors:")
    print()
    
    # Group by submodule
    cv_groups = {}
    for key in class_specific_keys:
        # Extract cv module (e.g., "model.23.cv2")
        parts = key.split('.')
        if len(parts) >= 3:
            cv_name = '.'.join(parts[:3])  # e.g., "model.23.cv2"
            if cv_name not in cv_groups:
                cv_groups[cv_name] = []
            cv_groups[cv_name].append(key)
    
    for cv_name in sorted(cv_groups.keys()):
        keys = cv_groups[cv_name]
        print(f"  {cv_name}:")
        
        # Check if dimensions depend on class count
        for key in keys[:2]:
            shape = state_dict[key].shape
            
            # Check if any dimension matches class count
            nc_dependent = model.model.nc in shape or (model.model.nc * 4) in shape
            marker = "⚠️ NC-DEPENDENT" if nc_dependent else ""
            
            print(f"    {key:60s} {str(shape):20s} {marker}")
        
        if len(keys) > 2:
            print(f"    ... {len(keys)-2} more weights")
        print()
    
    print(f"{'='*80}")
    print("Transfer Learning Guide")
    print(f"{'='*80}\n")
    
    print("To adapt this model to a different number of classes:")
    print()
    print("✓ PRESERVE (transfer these layers):")
    print("  - Layers 0-22: Backbone and neck")
    print("  - All weights with 'model.0.' through 'model.22.'")
    print()
    print("🔴 REINITIALIZE (skip these layers):")
    print("  - Layer 23: Detection/segmentation head")
    print("  - All weights with 'model.23.*'")
    print("  - Specifically:")
    print("    - model.23.cv2.*  (classification)")
    print("    - model.23.cv3.*  (box/mask prediction)")
    print("    - model.23.cv4.*  (additional heads)")
    print()
    print("This preserves learned features while adapting to new classes!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Inspect YOLO model architecture and identify class-specific layers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python inspect_model.py ~/models/best.pt
  
This tool shows:
  - Model architecture breakdown
  - Which layers are class-specific
  - Weight tensor names and shapes
  - Transfer learning guidance
        """
    )
    
    parser.add_argument('model_path', type=str,
                        help='Path to YOLO model (.pt file)')
    
    args = parser.parse_args()
    
    try:
        inspect_yolo_model(args.model_path)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())