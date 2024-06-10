import cv2
import os
import numpy as np

# print(os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'])

black = np.zeros((400,400,3),np.uint8)
# image = cv2.imread("wallpaper.png")
cv2.imshow("A wallpaper",black)
cv2.waitKey(0)