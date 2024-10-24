import cv2
import numpy as np


class Camera:
    def __init__(self, mtx, dist) -> None:
        self.mtx = mtx
        self.dist = dist
        self.pose = np.eye(4)
        self.projection = np.matmul(self.mtx, self.pose[:3, :])
        return

    def setPose(self, pose: np.ndarray) -> None:
        self.pose = pose
        self.projection = np.matmul(self.mtx, self.pose[:3, :])
        return


class Triangulation:
    def __init__(self, cam1: Camera, cam2: Camera) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.ArucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.cam1 = cam1
        self.cam2 = cam2
        self.relativePose = None
        return

    def getCameraPositionsStream(self, cap1: cv2.VideoCapture, cap2: cv2.VideoCapture, markerWidth: int) -> bool:
        while True:
            # read in images
            grabbed1, frame1 = cap1.read()
            grabbed2, frame2 = cap2.read()
            # detect markers
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            (corners1, ids1, rejectedImgPoints) = self.ArucoDetector.detectMarkers(gray1)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            (corners2, ids2, rejectedImgPoints) = self.ArucoDetector.detectMarkers(gray2)
            # if markers are found in both images
            if ids1 is not None and ids2 is not None:
                self.calcPoses(corners1, corners2, markerWidth)
                return True

    def getCameraPositionsImage(self, frame1: np.ndarray, frame2: np.ndarray, markerWidth: int) -> bool:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        (corners1, ids1, rejectedImgPoints) = self.ArucoDetector.detectMarkers(gray1)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        (corners2, ids2, rejectedImgPoints) = self.ArucoDetector.detectMarkers(gray2)
        if ids1 is not None and ids2 is not None:
            self.calcPoses(corners1, corners2, markerWidth)
            return True
        return False

    def calcPoses(self, corners1: np.ndarray, corners2: np.ndarray, markerWidth) -> None:
        #calculate mean x of the corners
        mean1 = np.mean(corners1[0][0][:,0])
        mean2 = np.mean(corners2[0][0][:,0])

        if mean1 > mean2:
            self.cam1, self.cam2 = self.cam2, self.cam1
        rvec1, tvec1, markerpos1 = cv2.aruco.estimatePoseSingleMarkers(
            corners1[0], markerWidth, self.cam1.mtx, self.cam1.dist)
        R1 = cv2.Rodrigues(rvec1)[0]
        rvec2, tvec2, markerpos2 = cv2.aruco.estimatePoseSingleMarkers(
            corners2[0], markerWidth, self.cam2.mtx, self.cam2.dist)
        R2 = cv2.Rodrigues(rvec2)[0] 
        # get the pose of the marker in each camera frame
        pose1 = np.eye(4)
        pose1[:3, :3] = R1
        pose1[:3, 3] = tvec1
        pose2 = np.eye(4)
        pose2[:3, :3] = R2
        pose2[:3, 3] = tvec2
        self.cam1.setPose(pose1)
        self.cam2.setPose(pose2)
        # get the relative pose of the two cameras
        self.relativePose = np.matmul(pose1, np.linalg.inv(pose2))
        return

    def get3dPoint(self, point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
        # get the 3d point from the two 2d points
        undistort1 = cv2.undistortImagePoints(
            point1, self.cam1.mtx, self.cam1.dist)
        undistort2 = cv2.undistortImagePoints(
            point2, self.cam2.mtx, self.cam2.dist)
        position = cv2.triangulatePoints(
            self.cam1.projection, self.cam2.projection, undistort1, undistort2)
        return position
