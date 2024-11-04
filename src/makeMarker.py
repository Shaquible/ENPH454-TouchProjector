"""
Code Authored by Keegan Kelly
"""
import cv2
import numpy as np
import sys

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
tagOut = np.ones((1000, 1000, 1), dtype='uint8')*0.9999
tag = np.zeros((600, 600, 1), dtype='uint8')
# args, dict, id, sideLength, img variable, borderBits
cv2.aruco.generateImageMarker(arucoDict, 10, 600, tag, 1)
tagOut[200:800, 200:800] = tag
cv2.imshow("aruco10", tagOut)
cv2.waitKey(0)
cv2.moveWindow("aruco10", 100, 100)
cv2.waitKey(0)
cv2.imwrite("aruco10.png", tagOut)
