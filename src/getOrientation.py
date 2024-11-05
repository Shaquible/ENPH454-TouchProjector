from webcamStream import openStream
from triangulation import Triangulation, Camera
import cv2
import numpy as np
imHeight = 1080
imWidth = 1920
exposure = -8
markerWidth = 0.1586
npfile = np.load("cameraIntrinsics/IRCam1.npz")
mtx1 = npfile["mtx"]
dist1 = npfile["dist"]
npfile = np.load("cameraIntrinsics/IRCam2.npz")
mtx2 = npfile["mtx"]
dist2 = npfile["dist"]
cap1 = openStream(0, imHeight, imWidth, exposure=exposure)
cap2 = openStream(1, imHeight, imWidth, exposure=exposure)
# need to crop the images
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        break

tri = Triangulation(Camera(mtx1, dist1), Camera(mtx2, dist2))
tri.getCameraPositionsStream(cap1, cap2, markerWidth)
np.save("relativePose", tri.relativePose)