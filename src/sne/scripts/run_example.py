import os
from SNE import SNE
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


class dataset():
    def __init__(self):
        self.num_labels = 2

if __name__ == '__main__':

    # Store images for plotting later
    depth_images = []
    normal_ground_truth_images = []
    normal_images = []

    for i in range(1,7):
        # if you want to use your own data, please modify rgb_image, depth_image, camParam and use_size correspondingly.
        depth_image = np.load(f'DIODE_dataset/depth/depth_{i}.npy')
        # Remove the single channel dimension if present
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze(axis=2)
        oriHeight, oriWidth = depth_image.shape
        oriSize = (oriWidth, oriHeight)

        # Load ground truth normal image
        normal_gt = np.load(f'DIODE_dataset/normal_ground_truth/normal_gt_{i}.npy')
        # Remove single channel dimension if present
        if normal_gt.ndim == 4:
            normal_gt = normal_gt.squeeze()
        # Transpose if needed to get (H, W, 3) format
        if normal_gt.shape[0] == 3:
            normal_gt = np.transpose(normal_gt, [1, 2, 0])

        # resize image to enable sizes divide 32
        use_size = (1248, 384)

        # compute normal using SNE
        sne_model = SNE()
        camParam = torch.tensor([[7.215377e+02, 0.000000e+00, 6.095593e+02],
                                [0.000000e+00, 7.215377e+02, 1.728540e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=torch.float32)  # camera parameters
        normal = sne_model(torch.tensor(depth_image.astype(np.float32)/1000), camParam)
        normal_image = normal.cpu().numpy()
        normal_image = np.transpose(normal_image, [1, 2, 0])
        
        # Save as PNG (converted to uint8 in [0, 255] range for visualization)
        cv2.imwrite(os.path.join('images', f'normal_{i}.png'), cv2.cvtColor(255*(1+normal_image)/2, cv2.COLOR_RGB2BGR).astype(np.uint8))
        
        # Store in original float32 [-1, 1] range for accurate comparison with ground truth
        depth_images.append(depth_image)
        normal_ground_truth_images.append(normal_gt)
        normal_images.append(normal_image)
    
    print(f"Ground truth range: [{np.min(normal_ground_truth_images):.3f}, {np.max(normal_ground_truth_images):.3f}]")
    print(f"Computed normals range: [{np.min(normal_images):.3f}, {np.max(normal_images):.3f}]")
    
    # Plot depth, ground truth normals, and computed normals side by side
    fig, axes = plt.subplots(6, 3, figsize=(18, 24))
    
    for i in range(6):
        # Plot depth image
        axes[i, 0].imshow(depth_images[i], cmap='viridis')
        axes[i, 0].set_title(f'Depth {i+1}')
        axes[i, 0].axis('off')
        
        # Plot ground truth normal image
        # Both normals are in [-1, 1] range, convert to [0, 1] for display
        normal_gt_display = (normal_ground_truth_images[i] + 1) / 2
        # Clip to [0, 1] range to ensure consistency
        normal_gt_display = np.clip(normal_gt_display, 0, 1)
        axes[i, 1].imshow(normal_gt_display)
        axes[i, 1].set_title(f'Ground Truth Normal {i+1}')
        axes[i, 1].axis('off')
        
        # Plot computed normal image
        # Both normals are in [-1, 1] range, convert to [0, 1] for display
        normal_display = (normal_images[i] + 1) / 2
        # Clip to [0, 1] range to ensure consistency
        normal_display = np.clip(normal_display, 0, 1)
        axes[i, 2].imshow(normal_display)
        axes[i, 2].set_title(f'Computed Normal {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('images/depth_normal_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparison plot saved to 'images/depth_normal_comparison.png'")
