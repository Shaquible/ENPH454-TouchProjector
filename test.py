import numpy as np

npfile = np.load("cameraIntrinsics/IRCam2Visible.npz")
print(npfile["mtx"])
npfile = np.load("cameraIntrinsics/IRCam2.npz")
print(npfile["mtx"])