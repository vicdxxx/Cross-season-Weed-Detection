import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import cv2
path = r'D:\Dataset\ruler.png'
im=cv2.imread(path)
plt.imshow(im)
plt.show()