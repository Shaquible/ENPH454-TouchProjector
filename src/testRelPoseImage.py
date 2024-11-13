import triangulationCharuco as tri
import cv2
import numpy as np

# Load the camera intrinsics
npfile = np.load("cameraIntrinsics/Cam1Vis.npz")
mtxVis = npfile["mtx"]
distVis = npfile["dist"]
npfile = np.load("cameraIntrinsics/Cam1IR.npz")
mtxIR = npfile["mtx"]
distIR = npfile["dist"]
cam1 = tri.Camera(np.copy(mtxIR), np.copy(distIR),
                  np.copy(mtxVis), np.copy(distVis))
npfile = np.load("cameraIntrinsics/Cam2Vis.npz")
mtxVis = npfile["mtx"]
distVis = npfile["dist"]
npfile = np.load("cameraIntrinsics/Cam2IR.npz")
mtxIR = npfile["mtx"]
distIR = npfile["dist"]
cam2 = tri.Camera(np.copy(mtxIR), np.copy(distIR),
                  np.copy(mtxVis), np.copy(distVis))
triangulation = tri.Triangulation(cam1, cam2)
cam1IR = cv2.imread("cam1IRPose.png")
cam1Vis = cv2.imread("Cam1VisPose.jpg")
cam2IR = cv2.imread("cam2IRPose.png")
cam2Vis = cv2.imread("Cam2VisPose.jpg")
triangulation.getCameraPositionsImage(cam1IR, cam2IR, IR=True)
triangulation.getCameraPositionsImage(cam1Vis, cam2Vis, IR=False)
print(triangulation.cam1.irPose)
print(triangulation.cam1.visPose)
print(triangulation.cam2.irPose)
print(triangulation.cam2.visPose)
# undistort the images
cam1IR = cv2.undistort(cam1IR, cam1.irMtx, cam1.irDist)
cam1Vis = cv2.undistort(cam1Vis, cam1.visMtx, cam1.visDist)
cam2IR = cv2.undistort(cam2IR, cam2.irMtx, cam2.irDist)
cam2Vis = cv2.undistort(cam2Vis, cam2.visMtx, cam2.visDist)
b, g, r = cv2.split(cam1Vis)
cv2.imwrite("Cam1VisPoseGray.png", b)
