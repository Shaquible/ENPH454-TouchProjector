"""
Code Authored by Keegan Kelly
"""
import cv2
import numpy as np
import sys

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
tagOut = np.ones((1800, 3900, 3), dtype='uint8')*255
tag = np.zeros((450, 450, 1), dtype='uint8')
#make a 5 by 12 grid of aruco markers
for i in range(3):
    for j in range(7):
        cv2.aruco.generateImageMarker(arucoDict, i*7+j, 450, tag, 1)
        xstart = 75+525*i
        ystart = 75+525*j
        tagOut[xstart: xstart + 450, ystart: ystart + 450, :] = tag
print(tagOut.shape)
cv2.imwrite("arucoGrid.png", tagOut)
# im = cv2.imread("aruco10.png")
# cv2.imshow("test", im)
# cv2.waitKey(0)
