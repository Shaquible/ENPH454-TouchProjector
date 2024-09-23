from handTracker import HandTracker
import cv2
import numpy as np
cap1 = cv2.VideoCapture(0)

cap2 = cv2.VideoCapture(1)
#turn off auto focus for both cameras and set focus to 0
cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap1.set(cv2.CAP_PROP_FOCUS, 0)
cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap2.set(cv2.CAP_PROP_FOCUS, 0)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
parameters =  cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

markerWidth = 0.1586
npfile = np.load("cameraIntrinsics/U4-2159.npz")
mtx1 = npfile["mtx"]
dist1 = npfile["dist"]
npfile = np.load("cameraIntrinsics/U4-2166.npz")
mtx2 = npfile["mtx"]
dist2 = npfile["dist"]

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    # cv2.imshow("frame1", frame1)
    # cv2.imshow("frame2", frame2)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    (corners1, ids1, rejectedImgPoints) = detector.detectMarkers(gray1)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    (corners2, ids2, rejectedImgPoints) = detector.detectMarkers(gray2)
    if ids1 is not None and ids2 is not None:
        #save frame 1 and frame 2
        cv2.imwrite("frame1.jpg", frame1)
        cv2.imwrite("frame2.jpg", frame2)
        R1, T1, markerpos1 = cv2.aruco.estimatePoseSingleMarkers(corners1[0], markerWidth, mtx1, dist1)
        R2, T2, markerpos2 = cv2.aruco.estimatePoseSingleMarkers(corners2[0], markerWidth, mtx2, dist2)
        R1 = R1[0][0]
        T1 = T1[0][0]
        R2 = R2[0][0]
        T2 = T2[0][0]
        print(T1, T2)
        Rod1 = cv2.Rodrigues(R1)[0]
        position = np.matmul(Rod1, T2+T1)
        print(position)
        break
        # calculates rotation matrix from the rotation vector
    if not ret1 or not ret2:
        break

cap1.release()
cap2.release()