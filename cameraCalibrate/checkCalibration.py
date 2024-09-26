import cv2
import numpy as np


npfile = np.load("wideAngleCalibration.npz")
mtx1 = npfile["mtx"]
dist1 = npfile["dist"]
#undistort the image
img = cv2.imread("chessPic120.png")
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx1, dist1, (w, h), 1, (w, h))
dst = cv2.undistort(img, mtx1, dist1, None, newcameramtx)
#show the image
cv2.imshow("img", img)
cv2.imshow("dst", dst)
cv2.waitKey(0)