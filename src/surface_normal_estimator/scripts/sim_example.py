import os
from SNE import SNE
import torch
import numpy as np
import matplotlib.pyplot as plt


class dataset():
    def __init__(self):
        self.num_labels = 2

if __name__ == '__main__':

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the frame_000 dataset
    depth_dir = os.path.join(script_dir, 'images', 'frame_000')

    # if you want to use your own data, please modify rgb_image, depth_image, camParam and use_size correspondingly.
    depth_image = np.load(os.path.join(depth_dir, 'depth.npy'))
    # Save depth image as CSV file
    # csv_path = os.path.join(depth_dir, 'depth.csv')
    # np.savetxt(csv_path, depth_image, delimiter=',')
    # print(f"Depth image saved to: {csv_path}")

    # Remove the single channel dimension if present
    if depth_image.ndim == 3:
        depth_image = depth_image.squeeze(axis=2)
    oriHeight, oriWidth = depth_image.shape
    oriSize = (oriWidth, oriHeight)

    # Convert depth to inverse depth
    inverse_depth_image = 1.0 / (depth_image + 1e-8)

    # Load ground truth normal image
    normal_gt = np.load(os.path.join(depth_dir, 'normals.npy'))
    print("normal_gt shape:", normal_gt.shape)

    # Extract only first 3 channels (XYZ) if there's an alpha channel
    if normal_gt.shape[2] == 4:
        print("Ground truth has 4 channels, extracting first 3 (XYZ)")
        normal_gt = normal_gt[:, :, :3]
    print("normal_gt shape after channel extraction:", normal_gt.shape)

    # Save ground truth normals as CSV file
    normal_gt_reshaped = normal_gt.reshape(-1, 3)
    csv_path_gt = os.path.join(depth_dir, 'ground_truth_normals.csv')
    np.savetxt(csv_path_gt, normal_gt_reshaped, delimiter=',', header='nx,ny,nz', comments='')
    print(f"Ground truth normals saved to: {csv_path_gt}")

    # compute normal using SNE
    sne_model = SNE()
    
    # Camera intrinsics from the fisheye/pinhole camera
    # focal lengths (fx, fy) and optical center (cx, cy)
    fx = 731.78788
    fy = 731.78788
    cx = 970.94244
    cy = 600.37482
    
    # Construct camera parameter matrix in standard format:
    camParam = torch.tensor([[fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0]], dtype=torch.float32)
    
    normal = sne_model(torch.tensor(inverse_depth_image.astype(np.float32)), camParam)
    
    normal_image = normal.cpu().numpy()
    print("Computed normal image shape:", normal_image.shape)
    normal_image = np.transpose(normal_image, [1, 2, 0])

    # Save normal image as CSV file
    # Reshape to 2D array where each row is a pixel's normal vector (x, y, z)
    normal_reshaped = normal_image.reshape(-1, 3)
    csv_path = os.path.join(depth_dir, 'computed_normals.csv')
    np.savetxt(csv_path, normal_reshaped, delimiter=',', header='nx,ny,nz', comments='')
    print(f"Normal image saved to: {csv_path}")


    # Plot ground truth vs computed normals side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot ground truth normal image
    # Convert from [-1, 1] range to [0, 1] for display
    normal_gt_display = (normal_gt + 1) / 2
    axes[0].imshow(normal_gt_display)
    axes[0].set_title('Ground Truth Normals')
    axes[0].axis('off')
    
    # Plot computed normal image
    # Convert from [-1, 1] range to [0, 1] for display
    normal_display = (normal_image + 1) / 2
    axes[1].imshow(normal_display)
    axes[1].set_title('Computed Normals (SNE)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    