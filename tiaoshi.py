import matplotlib.pyplot as plt
import skimage.io
import numpy as np
import os
from glob import glob
import cv2

# data = glob(os.path.join('D:/test11111111111111/SRGAN-master/SRGAN-master/Target_cifar10','number000.png'))
img = cv2.imread('D:/test11111111111111/SRGAN-master/SRGAN-master/Target_cifar10/number000.png')
img = np.array(img).astype(np.float32)
print(img.shape)
img_resize = img.resize(256, 256, 3)

# plt.figure("Image") # 图像窗口名称
# plt.imshow(img)
# plt.show()