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

    # Path to the frame_005 dataset
    depth_dir = os.path.join(script_dir, 'images', 'frame_006')

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
    
    normal = sne_model(torch.tensor(depth_image.astype(np.float32)), camParam)
    
    normal_image = normal.cpu().numpy()
    normal_image = np.transpose(normal_image, [1, 2, 0])
    print("Computed normal image shape:", normal_image.shape)

    # Save normal image as CSV file
    # Reshape to 2D array where each row is a pixel's normal vector (x, y, z)
    normal_reshaped = normal_image.reshape(-1, 3)
    csv_path = os.path.join(depth_dir, 'computed_normals.csv')
    np.savetxt(csv_path, normal_reshaped, delimiter=',', header='nx,ny,nz', comments='')
    print(f"Normal image saved to: {csv_path}")


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


    # 3D Visualization of Surface Normals
    # normal_image: computed normals, shape (360, 640, 3), values in [-1,1]
    # normal_gt: ground truth normals, shape (360, 640, 3), values in [-1,1]

    # --- Parameters ---
    patch_size = 100  # number of vectors along each axis
    scale = 0.5       # length of arrows for visibility
    offset = 50      # offset in the original image to take the patch from
    # --- Slice or downsample normals ---
    # You can take a patch offset by 50 pixels in both x and y:
    normals_image_subset = normal_image[offset:offset+patch_size, offset:offset+patch_size, :]
    normals_gt_subset = normal_gt[offset:offset+patch_size, offset:offset+patch_size, :]

    # Normalize the surface normal vectors
    norm_img = np.linalg.norm(normals_image_subset, axis=2, keepdims=True)
    normals_image_subset = normals_image_subset / (norm_img + 1e-8)
    
    norm_gt = np.linalg.norm(normals_gt_subset, axis=2, keepdims=True)
    normals_gt_subset = normals_gt_subset / (norm_gt + 1e-8)

    # --- Create grid for starting points ---
    X, Y = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    Z = np.zeros_like(X)

    # --- Extract vector components ---
    U_img, V_img, W_img = normals_image_subset[:, :, 0], normals_image_subset[:, :, 1], normals_image_subset[:, :, 2]
    U_gt, V_gt, W_gt = normals_gt_subset[:, :, 0], normals_gt_subset[:, :, 1], normals_gt_subset[:, :, 2]

    # --- Optional: color coding based on Z-component for visibility ---
    colors_img = (W_img - W_img.min()) / (W_img.max() - W_img.min())
    colors_gt = (W_gt - W_gt.min()) / (W_gt.max() - W_gt.min())

    # --- Plot with normal maps underneath ---
    fig = plt.figure(figsize=(16, 14))
    
    # Compute unified color scaling based on combined Z-components
    W_combined = np.concatenate([W_img.flatten(), W_gt.flatten()])
    W_min, W_max = W_combined.min(), W_combined.max()
    
    # Normalize colors using the same scale for both plots
    colors_img = (W_img - W_min) / (W_max - W_min + 1e-8)
    colors_gt = (W_gt - W_min) / (W_max - W_min + 1e-8)
    
    # Computed normals plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.quiver(X, Y, Z, U_img, V_img, W_img, length=scale, normalize=True, color=plt.cm.viridis(colors_img.flatten()))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Computed Normals (SNE)')
    ax1.set_xlim([0, patch_size])
    ax1.set_ylim([0, patch_size])
    ax1.set_zlim([-1, 1])

    # Ground truth normals plot
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.quiver(X, Y, Z, U_gt, V_gt, W_gt, length=scale, normalize=True, color=plt.cm.viridis(colors_gt.flatten()))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Ground Truth Normals')
    ax2.set_xlim([0, patch_size])
    ax2.set_ylim([0, patch_size])
    ax2.set_zlim([-1, 1])

    # Computed normal map visualization (2D image)
    ax3 = fig.add_subplot(223)
    # Convert normals from [-1, 1] to [0, 1] for display
    normals_image_display = (normals_image_subset + 1) / 2
    # Rotate 180 degrees
    #normals_image_display = np.rot90(normals_image_display, k=2)
    ax3.imshow(normals_image_display)
    ax3.set_title('Computed Normal Map (Patch)')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')

    # Ground truth normal map visualization (2D image)
    ax4 = fig.add_subplot(224)
    # Convert normals from [-1, 1] to [0, 1] for display
    normals_gt_display = (normals_gt_subset + 1) / 2
    # Rotate 180 degrees
    #normals_gt_display = np.rot90(normals_gt_display, k=2)
    ax4.imshow(normals_gt_display)
    ax4.set_title('Ground Truth Normal Map (Patch)')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')

    plt.tight_layout()
    plt.show()

    # Spherical Coordinates Visualization
    # Convert normals to spherical coordinates (theta, phi)
    # theta: azimuthal angle (0 to 2π)
    # phi: polar angle (0 to π)

    # def normals_to_spherical(normals):
    #     """Convert normal vectors (x, y, z) to spherical coordinates (theta, phi)"""
    #     # Normalize normals
    #     norms = np.linalg.norm(normals, axis=2, keepdims=True)
    #     normals_norm = normals / (norms + 1e-8)
        
    #     x = normals_norm[:, :, 0]
    #     y = normals_norm[:, :, 1]
    #     z = normals_norm[:, :, 2]
        
    #     # Phi: polar angle from z-axis (0 to π)
    #     phi = np.arccos(np.clip(z, -1, 1))
        
    #     # Theta: azimuthal angle in xy-plane (0 to 2π)
    #     theta = np.arctan2(y, x)
    #     # Convert to [0, 2π] range
    #     theta = np.where(theta < 0, theta + 2*np.pi, theta)
        
    #     return theta, phi

    # # Convert computed and ground truth normals to spherical coordinates
    # theta_computed, phi_computed = normals_to_spherical(normal_image)
    # theta_gt, phi_gt = normals_to_spherical(normal_gt)

    # # Flatten for plotting
    # theta_computed_flat = theta_computed.flatten()
    # phi_computed_flat = phi_computed.flatten()
    # theta_gt_flat = theta_gt.flatten()
    # phi_gt_flat = phi_gt.flatten()

    # print(f"Computed normals: {len(theta_computed_flat)} points")
    # print(f"Ground truth normals: {len(theta_gt_flat)} points")
    
    # # Create spherical coordinate plots
    # fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # # Computed normals in spherical coordinates
    # scatter1 = axes[0].scatter(theta_computed_flat, phi_computed_flat, 
    #                           c=phi_computed_flat, cmap='viridis', 
    #                           s=1, alpha=0.5)
    # axes[0].set_xlabel('Theta (azimuthal angle) [radians]')
    # axes[0].set_ylabel('Phi (polar angle) [radians]')
    # axes[0].set_title('Computed Normals in Spherical Coordinates')
    # axes[0].set_xlim([0, 2*np.pi])
    # axes[0].set_ylim([0, np.pi])
    # axes[0].grid(True, alpha=0.3)
    # plt.colorbar(scatter1, ax=axes[0], label='Phi (polar angle)')

    # # Ground truth normals in spherical coordinates
    # scatter2 = axes[1].scatter(theta_gt_flat, phi_gt_flat, 
    #                           c=phi_gt_flat, cmap='viridis', 
    #                           s=1, alpha=0.5)
    # axes[1].set_xlabel('Theta (azimuthal angle) [radians]')
    # axes[1].set_ylabel('Phi (polar angle) [radians]')
    # axes[1].set_title('Ground Truth Normals in Spherical Coordinates')
    # axes[1].set_xlim([0, 2*np.pi])
    # axes[1].set_ylim([0, np.pi])
    # axes[1].grid(True, alpha=0.3)
    # plt.colorbar(scatter2, ax=axes[1], label='Phi (polar angle)')

    # plt.tight_layout()
    # plt.show()

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
    
    