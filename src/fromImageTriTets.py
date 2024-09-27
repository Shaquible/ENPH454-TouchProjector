from handTracker import HandTracker
import cv2
import numpy as np
import time
import mediapipe as mp


def detectHands(frame):
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks


mp_hands = mp.solutions.hands

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

markerWidth = 0.1586
npfile = np.load("cameraIntrinsics/cam128-2.npz")
mtx1 = npfile["mtx"]
dist1 = npfile["dist"]
npfile = np.load("cameraIntrinsics/cam127-2.npz")
mtx2 = npfile["mtx"]
dist2 = npfile["dist"]

# getting reference frames for the 2 cameras
frame1 = cv2.imread("handframe1.jpg")
frame2 = cv2.imread("handframe2.jpg")
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
    # draw the axis on the marker
    # cv2.drawFrameAxes(frame1, mtx1, dist1, rvec1, tvec1, markerWidth/2)
    # cv2.drawFrameAxes(frame2, mtx2, dist2, rvec2, tvec2, markerWidth/2)

    R2 = cv2.Rodrigues(rvec2)[0]
    # find the relative position of the two markers
    pose1 = np.eye(4)
    pose1[:3, :3] = R1
    pose1[:3, 3] = tvec1
    pose2 = np.eye(4)
    pose2[:3, :3] = R2
    pose2[:3, 3] = tvec2
    # pose2 = np.linalg.inv(pose2)
    relativePose = np.matmul(pose1, np.linalg.inv(pose2))
    # convert back to rvec and tvec
    relativeRvec = np.array(cv2.Rodrigues(relativePose[:3, :3])[0]).T[0]
    relativeTvec = relativePose[:3, 3]
    print("relativeRvec", relativeRvec)
    print("relativeTvec", relativeTvec)
# calculating projection matrix
# proj1 = np.matmul(mtx1, np.linalg.inv(pose1)[:3, :])
# proj2 = np.matmul(mtx2, np.linalg.inv(pose2)[:3, :])
proj1 = np.matmul(mtx1, pose1[:3, :])
proj2 = np.matmul(mtx2, pose2[:3, :])

hands1 = detectHands(frame1)
hands2 = detectHands(frame2)

if hands1 is not None and hands2 is not None:
    for hand_landmarks in hands1:
        idx1F1 = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*1920,
                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*1080])
        print(idx1F1)
    for hand_landmarks in hands2:
        idx1F2 = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*1920,
                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*1080])
        print(idx1F2)
    outputPoint = cv2.triangulatePoints(
        proj1, proj2, idx1F1, idx1F2)
    print(outputPoint[0], outputPoint[1], outputPoint[2])
