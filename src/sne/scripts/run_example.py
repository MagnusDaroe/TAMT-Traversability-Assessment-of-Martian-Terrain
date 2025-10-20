import os
from SNE import SNE
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2


class dataset():
    def __init__(self):
        self.num_labels = 2

if __name__ == '__main__':

    # if you want to use your own data, please modify rgb_image, depth_image, camParam and use_size correspondingly.
    depth_image = cv2.imread(r'images/depth_u16.png', cv2.IMREAD_ANYDEPTH)
    oriHeight, oriWidth = depth_image.shape
    oriSize = (oriWidth, oriHeight)

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
    cv2.imwrite(os.path.join('images', 'normal.png'), cv2.cvtColor(255*(1+normal_image)/2, cv2.COLOR_RGB2BGR).astype(np.uint8))
    normal_image = cv2.resize(normal_image, use_size)

    normal_image = transforms.ToTensor()(normal_image).unsqueeze(dim=0)