import cv2
import numpy as np
from picoControl import PicoControl
from PIL import ImageGrab
import time


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
        # calculate mean x of the corners
        mean1 = np.mean(corners1[0][0][:, 0])
        mean2 = np.mean(corners2[0][0][:, 0])

        if mean1 > mean2:
            self.cam1, self.cam2 = self.cam2, self.cam1
        rvec1, tvec1, markerpos1 = cv2.aruco.estimatePoseSingleMarkers(
            corners1[0], markerWidth, self.cam1.mtx, self.cam1.dist)

        R1 = cv2.Rodrigues(rvec1)[0]
        rvec2, tvec2, markerpos2 = cv2.aruco.estimatePoseSingleMarkers(
            corners2[0], markerWidth, self.cam2.mtx, self.cam2.dist)
        R2 = cv2.Rodrigues(rvec2)[0]
        print(tvec1, tvec2)
        print(rvec1, rvec2)
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
        position = position.reshape(4)
        return (position[:3]/position[3])

    def getProjectorPositionStream(self, cap1: cv2.VideoCapture, cap2: cv2.VideoCapture):
        camFlip = False
        # pico = PicoControl(0)
        # pico.setIRCutFilter(1)
        # figure out coordinates of this and add white boarder
        imS = cv2.imread("src/arucoQuad.png")
        cv2.imshow("Aruco", imS)
        # move window to top left
        cv2.moveWindow("Aruco", 0, 0)
        cv2.waitKey(1)
        time.sleep(2)
        screenShot = np.array(ImageGrab.grab())
        screenShot = cv2.cvtColor(screenShot, cv2.COLOR_BGR2GRAY)

        # screenShot = cv2.imread("screenShot.png")
        (screenShotCorners, idsScreenShot,
         rejectedImgPoints) = self.ArucoDetector.detectMarkers(screenShot)
        for corner, id in zip(screenShotCorners, idsScreenShot):
            if id == 10:
                screenTL = corner[0]
            elif id == 11:
                screenTR = corner[1]
            elif id == 12:
                screenBL = corner[3]
            elif id == 13:
                screenBR = corner[2]
        # do a screen grab of the display
        # find the marker in the screen shot to get pixel coordinates
        while True:
            # read in images
            grabbed1, frame1 = cap1.read()
            grabbed2, frame2 = cap2.read()
            gray1, r1, g1 = cv2.split(frame1)
            gray2, r2, g2 = cv2.split(frame2)
            # gray1 = cv2.imread("gray1.png")
            # gray2 = cv2.imread("gray2.png")
            # detect markers
            (corners1, ids1, rejectedImgPoints) = self.ArucoDetector.detectMarkers(gray1)
            (corners2, ids2, rejectedImgPoints) = self.ArucoDetector.detectMarkers(gray2)
            if len(ids1) == 4 and len(ids2) == 4:
                # figure out which cam is which
                mean1 = np.mean(corners1[0][0][:, 0])
                mean2 = np.mean(corners2[0][0][:, 0])
                if mean1 > mean2:
                    self.cam1, self.cam2 = self.cam2, self.cam1
                    camFlip = True
                for corner, id in zip(corners1, ids1):
                    if id == 10:
                        cam1TL = corner[0]
                    elif id == 11:
                        cam1TR = corner[1]
                    elif id == 12:
                        cam1BL = corner[3]
                    elif id == 13:
                        cam1BR = corner[2]
                for corner, id in zip(corners2, ids2):
                    if id == 10:
                        cam2TL = corner[0]
                    elif id == 11:
                        cam2TR = corner[1]
                    elif id == 12:
                        cam2BL = corner[3]
                    elif id == 13:
                        cam2BR = corner[2]

                TL = self.get3dPoint(cam1TL, cam2TL)
                TR = self.get3dPoint(cam1TR, cam2TR)
                BL = self.get3dPoint(cam1TR, cam2BL)
                pose1 = getTransformationMatrix(TL, TR, BL)
                print(pose1)
                pose2 = np.matmul(np.linalg.inv(self.relativePose), pose1)
                self.cam1.setPose(pose1)
                self.cam2.setPose(pose2)
                TL = self.get3dPoint(cam1TL, cam2TL)
                TR = self.get3dPoint(cam1TR, cam2TR)
                BL = self.get3dPoint(cam1BL, cam2BL)
                BR = self.get3dPoint(cam1BR, cam2BR)
                points1 = np.array([TL[:2], TR[:2], BL[:2], BR[:2]])
                points2 = np.array([screenTL, screenTR, screenBL, screenBR])
                matrix = cv2.getPerspectiveTransform(points1, points2)
                # pico.setIRCutFilter(0)
                cv2.destroyAllWindows()
                return matrix, camFlip


def getTransformationMatrix(TL, TR, BL):
    x = TR - TL
    y = BL - TL
    z = np.cross(x, y)
    y = np.cross(z, x)
    # normalize the orthonormal basis vectors
    x = x/np.linalg.norm(x)
    y = y/np.linalg.norm(y)
    z = z/np.linalg.norm(z)
    T = np.eye(4)
    # might need to subtract TL from the 3 rows one post said so, but cant tell till we test and fix the calibration
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = TL
    return T
