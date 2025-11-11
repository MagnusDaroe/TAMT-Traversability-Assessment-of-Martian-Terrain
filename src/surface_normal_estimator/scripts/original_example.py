import os
from SNE import SNE
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


class dataset():
    def __init__(self):
        self.num_labels = 2

if __name__ == '__main__':

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the depth image
    depth_path = os.path.join(script_dir, 'images', 'depth_u16.png')
    
    # Load depth image from PNG file
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    # Convert to float for processing
    depth_image = depth_image.astype(np.float32)
    depth_image /= 1000.0  # Convert from mm to meters
    # Save depth image as CSV file
    csv_path = os.path.join(script_dir, 'images', 'depth_original.csv')
    np.savetxt(csv_path, depth_image, delimiter=',')
    print(f"Depth image saved to: {csv_path}")
    print(f"Depth min: {depth_image.min():.2f}, max: {depth_image.max():.2f}, average: {depth_image.mean():.2f}")

    # Remove the single channel dimension if present
    if depth_image.ndim == 3:
        depth_image = depth_image.squeeze(axis=2)
    oriHeight, oriWidth = depth_image.shape
    oriSize = (oriWidth, oriHeight)

    # Convert depth to inverse depth
    inverse_depth_image = 1.0 / (depth_image + 1e-8)

    # compute normal using SNE
    sne_model = SNE()
    
    camParam = torch.tensor([[7.215377e+02, 0.000000e+00, 6.095593e+02],
                            [0.000000e+00, 7.215377e+02, 1.728540e+02],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=torch.float32)  # camera parameters
    
    normal = sne_model(torch.tensor(depth_image.astype(np.float32)), camParam)
    
    normal_image = normal.cpu().numpy()
    normal_image = np.transpose(normal_image, [1, 2, 0])

    # Plot computed normals
    plt.figure(figsize=(8, 6))
    
    # Convert from [-1, 1] range to [0, 1] for display
    normal_display = (normal_image + 1) / 2
    plt.imshow(normal_display)
    plt.title('Computed Normals (SNE)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    
    