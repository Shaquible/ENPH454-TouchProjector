import cv2
import numpy as np
from picoControl import PicoControl


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
        return position

    def getProjectorPositionStream(self, cap1: cv2.VideoCapture, cap2: cv2.VideoCapture, markerWidth: int) -> bool:
        pico = PicoControl(0)
        pico.setIRCutFilter(1)
        # figure out coordinates of this and add white boarder
        cv2.imshow("Aruco", cv2.imread("src/aruco10.png"))
        cv2.waitKey(0)
        # do a screen grab of the display
        # find the marker in the screen shot to get pixel coordinates
        while True:
            # read in images
            grabbed1, frame1 = cap1.read()
            grabbed2, frame2 = cap2.read()
            # detect markers
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            (corners1, ids1, rejectedImgPoints) = self.ArucoDetector.detectMarkers(gray1)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            (corners2, ids2, rejectedImgPoints) = self.ArucoDetector.detectMarkers(gray2)
            if ids1 is not None and ids2 is not None:
                pico.setIRCutFilter(0)
                cv2.destroyAllWindows()
                # figure out which cam is which and
                mean1 = np.mean(corners1[0][0][:, 0])
                mean2 = np.mean(corners2[0][0][:, 0])
                if mean1 > mean2:
                    self.cam1, self.cam2 = self.cam2, self.cam1
                corners1 = corners1[0]
                corners2 = corners2[0]
                points = np.zeros((4, 3))
                for i, corner1, corner2 in enumerate(zip(corners1, corners2)):
                    points[i] = self.get3dPoint(corner1, corner2)[:3]
                # markers  TL TR BR BL
                # markerWidth = np.mean(
                #     [np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]), np.linalg.norm(points[2] - points[3])])
                # rvec1, tvec1, markerpos1 = cv2.aruco.estimatePoseSingleMarkers(
                #     corners1[0], markerWidth, self.cam1.mtx, self.cam1.dist)

                # find rotation and translation of the marker from the 3d points
                # use the pixel coordinates of the marker and the length of the edges to extrapolate the TL and BR corners
                # get the rvec and tvec of the TL corner and the width and length of the projected image

                # return the rvec and tvec of the TL corner and the width and length of the projected image



    def createPlane(P1,P2, P3):
        v1 = P2 - P1
        v2 = P3 - P1

        # Calculate the normal vector using the cross product
        v3 = np.cross(v1, v2)



        # Normalize the normal vector (optional)
        v1Norm = v1 / np.linalg.norm(v1)
        v2Norm = v2 / np.linalg.norm(v2)
        v3Norm = v3 / np.linalg.norm(v3)

        return(v1Norm,v2Norm,v3Norm)

    def transform(center, v1,v2,v3):

        transform = [[v1[0],v1[1], v1[2], -center[0] ], [v2[0],v2[1], v2[2], -center[1] ],[v3[0],v3[1], v3[2], -center[2] ],[0,0, 0, 1 ]]
        return transform
    
    # def averagePlaneVectors(vectorArray):

    #     vectorArrayUse = abs(vectorArray)

    #     print(vectorArrayUse)

    #     xVectors = []
    #     yVectors = []


    #     zVectors = vectorArray[:][2]

    #     print(zVectors)



        
        

