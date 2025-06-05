import matplotlib 
from matplotlib import pyplot as plt
import cv2
import numpy as np
disp :np.ndarray = cv2.imread("disp/0048.pfm",cv2.IMREAD_UNCHANGED)
disp = ( 255 * (disp - np.min(disp)) / (np.max(disp) - np.min(disp))).astype(np.uint8)
plt.imsave("0048.png", disp, cmap='jet')