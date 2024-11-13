from webcamStream import openStream
from triangulationCharuco import Triangulation, Camera
import cv2
import numpy as np
imHeight = 1080
imWidth = 1920
exposure = -8
IR = True
npfile = np.load("cameraIntrinsics/Cam1IR.npz")
mtx1 = npfile["mtx"]
dist1 = npfile["dist"]
npfile = np.load("cameraIntrinsics/Cam2IR.npz")
mtx2 = npfile["mtx"]
dist2 = npfile["dist"]

npfile = np.load("cameraIntrinsics/Cam1Vis.npz")
mtx1Vis = npfile["mtx"]
dist1Vis = npfile["dist"]
npfile = np.load("cameraIntrinsics/Cam2Vis.npz")
mtx2Vis = npfile["mtx"]
dist2Vis = npfile["dist"]
cap1 = openStream(0, imHeight, imWidth, exposure=exposure)
cap2 = openStream(1, imHeight, imWidth, exposure=exposure)
# need to crop the images
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        break

tri = Triangulation(Camera(mtx1, dist1, mtx1Vis, dist1Vis),
                    Camera(mtx2, dist2, mtx2Vis, dist2Vis))
tri.getCameraPositionsStream(cap1, cap2, IR)
if IR:
    np.savez("cameraIntrinsics/IRPoses.npz",
             cam1Pose=tri.cam1.irPose, cam2Pose=tri.cam2.irPose)
else:
    np.savez("cameraIntrinsics/VisPoses.npz",
             cam1Pose=tri.cam1.visPose, cam2Pose=tri.cam2.visPose)
