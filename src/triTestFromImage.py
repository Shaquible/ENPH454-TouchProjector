from handTracker import HandTracker
import cv2
import numpy as np


ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

markerWidth = 0.159
npfile = np.load("cameraIntrinsics/U4-2159.npz")
mtx1 = npfile["mtx"]
dist1 = npfile["dist"]
npfile = np.load("cameraIntrinsics/U4-2166.npz")
mtx2 = npfile["mtx"]
dist2 = npfile["dist"]
# npfile = np.load("cameraIntrinsics/LogitechC920.npz")
# mtx1 = npfile["mtx"]
# dist1 = npfile["dist"]
# mtx2 = mtx1
# dist2 = dist1
frame1 = cv2.imread("frame2.jpg")
frame2 = cv2.imread("frame1.jpg")


gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
(corners1, ids1, rejectedImgPoints) = detector.detectMarkers(gray1)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
(corners2, ids2, rejectedImgPoints) = detector.detectMarkers(gray2)
if ids1 is not None and ids2 is not None:
    # find the same maker in each camera frame
    rvec1, tvec1, markerpos1 = cv2.aruco.estimatePoseSingleMarkers(
        corners1[0], markerWidth, mtx1, dist1)
    R1 = cv2.Rodrigues(rvec1)[0]
    rvec2, tvec2, markerpos2 = cv2.aruco.estimatePoseSingleMarkers(
        corners2[0], markerWidth, mtx2, dist2)
    R2 = cv2.Rodrigues(rvec2)[0]
    # find the relative position of the two markers
    pose1 = np.eye(4)
    pose1[:3, :3] = R1
    pose1[:3, 3] = tvec1
    pose2 = np.eye(4)
    pose2[:3, :3] = R2
    pose2[:3, 3] = tvec2
    pose2 = np.linalg.inv(pose2)
    relativePose = np.dot(pose2, pose1)
    # convert back to rvec and tvec
    relativeRvec = np.array(cv2.Rodrigues(relativePose[:3, :3])[0]).T[0]
    relativeTvec = relativePose[:3, 3]
    print("relativeRvec", relativeRvec)
    print("relativeTvec", relativeTvec)
