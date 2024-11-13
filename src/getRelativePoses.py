import numpy as np

npfile1 = np.load("cameraIntrinsics/IRPoses.npz")
npfile2 = np.load("cameraIntrinsics/VisPoses.npz")
cam1PoseIR = npfile1["cam1Pose"]
cam2PoseIR = npfile1["cam2Pose"]
cam1PoseVis = npfile2["cam1Pose"]
cam2PoseVis = npfile2["cam2Pose"]
print(cam1PoseIR)
print(cam2PoseIR)
print(cam1PoseVis)
print(cam2PoseVis)
relativePoseIR = np.matmul(cam1PoseIR, np.linalg.inv(cam2PoseIR))
relativePoseVis = np.matmul(cam1PoseVis, np.linalg.inv(cam2PoseVis))
cam1VisToIRPose = np.matmul(cam1PoseVis, np.linalg.inv(cam1PoseIR))

np.savez("cameraIntrinsics/relativePoses.npz", relativePoseIR=relativePoseIR,
         relativePoseVis=relativePoseVis, cam1VisToIRPose=cam1VisToIRPose)
print(relativePoseIR)
print(relativePoseVis)
print(cam1VisToIRPose)
print(np.linalg.norm(relativePoseIR[0:3, 3]))
print(np.linalg.norm(relativePoseVis[0:3, 3]))
print(np.linalg.norm(cam1VisToIRPose[0:3, 3]))
