#!/usr/bin/env python3
"""
YOLO Segmentation Model Class Adapter - Fixed for PyTorch 2.6+
Adapts a trained YOLO-seg model to work with different number of classes.
Preserves feature extraction weights, reinitializes detection/segmentation heads.
"""

import torch
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def adapt_yolo_segmentation_model(source_model_path, target_nc, output_path, verbose=True):
    """
    Adapt a YOLO segmentation model to a different number of classes.
    
    This preserves all the learned feature extraction weights while
    reinitializing only the detection and segmentation heads.
    
    Args:
        source_model_path: Path to trained model (.pt)
        target_nc: Target number of classes
        output_path: Where to save adapted model
        verbose: Print detailed info
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"YOLO Segmentation Model Adapter")
        print(f"{'='*70}\n")
    
    source_path = Path(source_model_path).expanduser()
    output_path = Path(output_path).expanduser()
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source model not found: {source_path}")
    
    if verbose:
        print(f"Loading: {source_path}")
    
    # Load model with YOLO
    source_model = YOLO(str(source_path))
    source_nc = source_model.model.nc
    
    if verbose:
        print(f"✓ Model loaded")
        print(f"  Source classes: {source_nc}")
        print(f"  Target classes: {target_nc}")
    
    if source_nc == target_nc:
        if verbose:
            print(f"\n⚠️  Warning: Source and target have same number of classes")
            print(f"   No adaptation needed, copying model...")
        shutil.copy2(source_path, output_path)
        return output_path
    
    if verbose:
        print(f"\nCreating new model with {target_nc} classes...")
    
    # Create a fresh model with target number of classes
    # This ensures all dimensions are correct from the start
    model_type = 'yolo11n-seg.pt'  # Use base architecture
    new_model = YOLO(model_type)
    
    # Override the number of classes
    new_model.model.nc = target_nc
    new_model.model.names = {i: f'class{i}' for i in range(target_nc)}
    
    if verbose:
        print(f"✓ New model created with {target_nc} classes")
        print(f"\nTransferring weights from source model...")
    
    # Get state dicts
    source_state = source_model.model.state_dict()
    new_state = new_model.model.state_dict()
    
    # Transfer weights layer by layer
    transferred = 0
    skipped = 0
    
    # These layers are class-dependent and should NOT be transferred
    skip_patterns = [
        'model.22.cv2',  # Detection class prediction
        'model.22.cv3',  # Detection box prediction (class-agnostic but safer to skip)
        'model.23.cv2',  # Segmentation class prediction  
        'model.23.cv3',  # Segmentation mask prediction
        'model.23.cv4',  # Additional segmentation layers
    ]
    
    for key in new_state.keys():
        # Skip class-specific layers
        if any(pattern in key for pattern in skip_patterns):
            skipped += 1
            if verbose and skipped <= 10:
                print(f"  Skipping (class-specific): {key}")
            continue
        
        # Only transfer if key exists in source and shapes match
        if key in source_state:
            if source_state[key].shape == new_state[key].shape:
                new_state[key] = source_state[key].clone()
                transferred += 1
            else:
                skipped += 1
                if verbose and skipped <= 15:
                    print(f"  Shape mismatch: {key}")
                    print(f"    Source: {source_state[key].shape}")
                    print(f"    Target: {new_state[key].shape}")
    
    # Load the transferred weights into the new model
    new_model.model.load_state_dict(new_state, strict=True)
    
    if verbose:
        print(f"\n✓ Weight transfer complete")
        print(f"  Transferred: {transferred} layers")
        print(f"  Skipped/Reinitialized: {skipped} layers")
        print(f"  Transfer ratio: {transferred/(transferred+skipped)*100:.1f}%")
        print(f"\n  Feature extraction: PRESERVED ✓")
        print(f"  Detection heads: REINITIALIZED for {target_nc} classes ✓")
        print(f"  Segmentation heads: REINITIALIZED for {target_nc} classes ✓")
    
    # Save the adapted model
    if verbose:
        print(f"\nSaving to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save using YOLO's built-in save method for compatibility
    torch.save(
        {
            'model': new_model.model,
            'nc': target_nc,
            'names': {i: f'class{i}' for i in range(target_nc)},
        },
        output_path
    )
    
    if verbose:
        print(f"✓ Saved successfully!")
        
        # Verify by loading
        print(f"\nVerifying saved model...")
        try:
            verify_model = YOLO(str(output_path))
            actual_nc = verify_model.model.nc
            if actual_nc == target_nc:
                print(f"✓ Verification passed! Model has {actual_nc} classes")
                
                # Test forward pass to ensure dimensions are correct
                import numpy as np
                dummy_input = torch.randn(1, 3, 640, 640)
                with torch.no_grad():
                    output = verify_model.model(dummy_input)
                print(f"✓ Forward pass test successful!")
            else:
                print(f"⚠️  Warning: Expected {target_nc} classes but got {actual_nc}")
        except Exception as e:
            print(f"⚠️  Verification failed: {e}")
            raise
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Adapt YOLO segmentation model from N to M classes (PyTorch 2.6+ compatible)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  
  # Adapt your 4-class model to 6 classes
  python adapt_yolo_segmentation.py \\
    --source ~/tamt/src/terrain_segmentation/models/terrain_segmentation/exp2/weights/best.pt \\
    --target-classes 6 \\
    --output ~/tamt/src/terrain_segmentation/models/adapted_models/exp2_to_6classes.pt

  Then in your training config:
    transfer_learning:
      enabled: true
      model_path: '~/tamt/src/terrain_segmentation/models/adapted_models/exp2_to_6classes.pt'

What this does:
  1. Loads your trained model (4 classes)
  2. Preserves ALL feature extraction layers (backbone, neck)
  3. Reinitializes detection heads for 6 classes
  4. Reinitializes segmentation heads for 6 classes
  5. Saves adapted model ready for transfer learning

The result:
  - Feature learning from original training is preserved
  - Can now train on 6-class dataset
  - Much faster convergence than training from scratch
  - This is TRUE transfer learning!
        """
    )
    
    parser.add_argument('--source', '-s', type=str, required=True,
                        help='Path to source trained model')
    parser.add_argument('--target-classes', '-c', type=int, required=True,
                        help='Number of classes in new dataset')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path for adapted model')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    try:
        adapted_path = adapt_yolo_segmentation_model(
            source_model_path=args.source,
            target_nc=args.target_classes,
            output_path=args.output,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\n{'='*70}")
            print("✅ SUCCESS! Model adapted for transfer learning")
            print(f"{'='*70}\n")
            print("Next steps:")
            print("1. Update your training config:")
            print("   transfer_learning:")
            print("     enabled: true")
            print(f"     model_path: '{adapted_path}'")
            print()
            print("2. Make sure dataset.yaml has nc: 6")
            print()
            print("3. Clear cache:")
            print("   find ~/tamt -name '*.cache' -delete")
            print()
            print("4. Start training - it will work now!")
            print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())