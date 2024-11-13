import numpy as np

npfile1 = np.load("cameraIntrinsics/IRPoses.npz")
npfile2 = np.load("cameraIntrinsics/VisPoses.npz")
cam1PoseIR = npfile1["cam1Pose"]
cam2PoseIR = npfile1["cam2Pose"]
cam1PoseVis = npfile2["cam1Pose"]
cam2PoseVis = npfile2["cam2Pose"]
relativePoseIR = np.matmul(np.linalg.inv(cam1PoseIR), cam2PoseIR)
relativePoseVis = np.matmul(np.linalg.inv(cam1PoseVis), cam2PoseVis)
cam1IRtoVisPose = np.matmul(np.linalg.inv(cam1PoseVis), cam1PoseIR)

np.savez("cameraIntrinsics/relativePoses.npz", relativePoseIR=relativePoseIR,
         relativePoseVis=relativePoseVis, cam1IRtoVisPose=cam1IRtoVisPose)
