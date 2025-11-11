import os
from SNE import SNE
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class dataset():
    def __init__(self):
        self.num_labels = 2

if __name__ == '__main__':

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the DIODE dataset (depth and normals)
    depth_dir = os.path.join(os.path.dirname(script_dir), 'DIODE_dataset')

    # Store images for plotting later
    depth_images = []
    normal_ground_truth_images = []
    normal_images = []

    for i in range(1,7):
        # if you want to use your own data, please modify rgb_image, depth_image, camParam and use_size correspondingly.
        depth_image = np.load(os.path.join(depth_dir, 'depth', f'depth_{i}.npy'))

        # Remove the single channel dimension if present
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze(axis=2)
        oriHeight, oriWidth = depth_image.shape
        oriSize = (oriWidth, oriHeight)

        # Load ground truth normal image
        normal_gt = np.load(os.path.join(depth_dir, 'normal_ground_truth', f'normal_gt_{i}.npy'))
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
        print("camParam:", camParam[0,0])
        normal = sne_model(torch.tensor(depth_image.astype(np.float32)/1000), camParam)
        
        print("normal image shape:", normal.shape)
        normal_image = normal.cpu().numpy()
        normal_image = np.squeeze(normal_image)
        normal_image = np.transpose(normal_image, [1, 2, 0])
        
        # # Apply 180 degree rotation about X-axis and Y-axis
        rotation_x = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        rotation_y = Rotation.from_euler('y', 180, degrees=True).as_matrix()
        normal_image = (rotation_y @ rotation_x @ normal_image.reshape(-1, 3).T).T.reshape(oriHeight, oriWidth, 3)

        # Save as PNG (converted to uint8 in [0, 255] range for visualization)
        # images_dir = os.path.join(script_dir, 'images')
        # os.makedirs(images_dir, exist_ok=True)
        # cv2.imwrite(os.path.join(images_dir, f'normal_{i}.png'), cv2.cvtColor((255*(1+normal_image)/2).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # Store in original float32 [-1, 1] range for accurate comparison with ground truth
        depth_images.append(depth_image)
        normal_ground_truth_images.append(normal_gt)
        normal_images.append(normal_image)

        def compute_aae(gt, pred, eps=1e-8):
            # compute norms and validity mask
            n1, n2 = np.linalg.norm(gt, 2, 2), np.linalg.norm(pred, 2, 2)
            valid = (n1 > eps) & (n2 > eps)
            
            # normalize valid normals
            gt_n = np.zeros_like(gt); pred_n = np.zeros_like(pred)
            gt_n[valid] = gt[valid] / n1[valid, None]
            pred_n[valid] = pred[valid] / n2[valid, None]
            
            # cosine similarity (use absolute for unoriented)
            dot = np.sum(gt_n * pred_n, 2)
            dot = np.clip(np.abs(dot), 0, 1)
            
            # angular error in degrees, ignoring invalid pixels
            ang = np.degrees(np.arccos(dot))
            ang[~valid] = np.nan
            return np.nanmean(ang)
    
        aae = compute_aae(normal_gt, normal_image)
        print(f"Average Angular Error (unoriented): {aae:.2f}°")
    
    print(f"Ground truth range: [{np.min(normal_ground_truth_images):.3f}, {np.max(normal_ground_truth_images):.3f}]")
    print(f"Computed normals range: [{np.min(normal_images):.3f}, {np.max(normal_images):.3f}]")
    


    # Plot depth, ground truth normals, and computed normals side by side
    n = len(depth_images)-2
    if n == 0:
        print("No images to plot.")
    else:
        cols = 3
        rows = n
        # size each subplot roughly 6x4 inches
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))

        # Ensure axes is 2D array for consistent indexing
        axes = np.atleast_2d(axes)

        for i in range(n):
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
    #plt.savefig(os.path.join(images_dir, 'depth_normal_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    #print(f"Comparison plot saved to '{os.path.join(images_dir, 'depth_normal_comparison.png')}'")