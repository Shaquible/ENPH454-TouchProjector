import cv2
import numpy as np
from picoControl import PicoControl
from PIL import ImageGrab
import time
import scipy.optimize as opt


class Camera:
    def __init__(self, irMtx, irDist, visMtx, visDist) -> None:
        self.irMtx = irMtx
        self.irDist = irDist
        self.visMtx = visMtx
        self.visDist = visDist
        self.visPose = np.eye(4)
        self.irPose = np.eye(4)
        self.visProjection = np.matmul(self.visMtx, self.visPose[:3, :])
        self.irProjection = np.matmul(self.irMtx, self.irPose[:3, :])
        return

    def setIRPose(self, pose: np.ndarray) -> None:
        self.irPose = pose
        self.irProjection = np.matmul(self.irMtx, self.irPose[:3, :])
        return

    def setVisPose(self, pose: np.ndarray) -> None:
        self.visPose = pose
        self.visProjection = np.matmul(self.visMtx, self.visPose[:3, :])
        return


class Triangulation:
    def __init__(self, cam1: Camera, cam2: Camera) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        parameters = cv2.aruco.DetectorParameters()
        parameters.minMarkerPerimeterRate = 0.1
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.charucoBoardPaper = cv2.aruco.CharucoBoard(
            (4, 6), 0.045125, 0.022625, dictionary)
        self.charucoBoard = cv2.aruco.CharucoBoard(
            (3, 7), 0.03, 0.02, dictionary)
        self.ArucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.cam1 = cam1
        self.cam2 = cam2
        self.cam1VisToIRPose = None
        self.relativePoseIR = None
        self.relativePoseIR = None
        return

    def getCameraPositionsStream(self, cap1: cv2.VideoCapture, cap2: cv2.VideoCapture, IR=True) -> bool:
        while True:
            # read in images
            grabbed1, frame1 = cap1.read()
            grabbed2, frame2 = cap2.read()
            if not grabbed1 or not grabbed2:
                continue
            # detect markers
            if IR:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, g, r = cv2.split(frame1)
                gray2, g, r = cv2.split(frame2)
            # gray1 = cv2.undistort(gray1, self.cam1.mtx, self.cam1.dist)
            (corners1, ids1, rejectedImgPoints1) = self.ArucoDetector.detectMarkers(gray1)

            # gray2 = cv2.undistort(gray2, self.cam2.mtx, self.cam2.dist)
            (corners2, ids2, rejectedImgPoints2) = self.ArucoDetector.detectMarkers(gray2)
            # if markers are found in both images
            if ids1 is not None and ids2 is not None:
                found = self.calcPoses(corners1, corners2, ids1, ids2, rejectedImgPoints1, rejectedImgPoints2,
                                       gray1, gray2, IR)

                if found:
                    return True

    def getCameraPositionsImage(self, frame1: np.ndarray, frame2: np.ndarray, IR=True) -> bool:
        if IR:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, g, r = cv2.split(frame1)
            gray2, g, r = cv2.split(frame2)
        cv2.imshow("cam1", gray1)
        cv2.imshow("cam2", gray2)
        cv2.waitKey(1)
        (corners1, ids1, rejectedImgPoints1) = self.ArucoDetector.detectMarkers(gray1)
        (corners2, ids2, rejectedImgPoints2) = self.ArucoDetector.detectMarkers(gray2)
        print(len(ids1), len(ids2))
        if ids1 is not None and ids2 is not None:
            found = self.calcPoses(corners1, corners2, ids1, ids2, rejectedImgPoints1, rejectedImgPoints2,
                                   gray1, gray2, IR)
            if found:
                return True
        return False

    def getCharucoCorner(self, corners: np.ndarray, ids: np.ndarray, rejectedCorners, im: np.ndarray, camNum, IR=True, board=None) -> np.ndarray:
        if board is None:
            board = self.charucoBoard
        cam = None
        if camNum == 1:
            cam = self.cam1
        elif camNum == 2:
            cam = self.cam2
        if IR and cam is not None:
            mtx = cam.irMtx
            dist = cam.irDist
        elif cam is not None:
            mtx = cam.visMtx
            dist = cam.visDist
        if cam is not None:
            corners, ids, rejected, recorvered = cv2.aruco.refineDetectedMarkers(
                im, board, corners, ids, rejectedCorners, cameraMatrix=mtx, distCoeffs=dist, errorCorrectionRate=3.0)
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, im, board, cameraMatrix=mtx, distCoeffs=dist)
        else:
            corners, ids, rejected, recorvered = cv2.aruco.refineDetectedMarkers(
                im, board, corners, ids, rejectedCorners)
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, im, board)
        if charuco_retval:
            return charuco_corners, charuco_ids
        return None, None

    def calcPoses(self, corners1: np.ndarray, corners2: np.ndarray, ids1, ids2, rejectedPoints1, rejectedPoints2, im1, im2, IR=True) -> None:
        charuco_corners1, charuco_ids1 = self.getCharucoCorner(
            corners1, ids1, rejectedPoints1, im1, 1, IR, self.charucoBoardPaper)
        if charuco_corners1 is not None:
            if IR:
                mtx = self.cam1.irMtx
                dist = self.cam1.irDist
            else:
                mtx = self.cam1.visMtx
                dist = self.cam1.visDist
            retval1, rvec1, tvec1 = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners1, charuco_ids1, self.charucoBoardPaper, mtx, dist, None, None)
        else:
            return False

        charuco_corners2, charuco_ids2 = self.getCharucoCorner(
            corners2, ids2, rejectedPoints2, im2, 2, IR, self.charucoBoardPaper)
        if charuco_corners2 is not None:
            if IR:
                mtx = self.cam2.irMtx
                dist = self.cam2.irDist
            else:
                mtx = self.cam2.visMtx
                dist = self.cam2.visDist
            retval2, rvec2, tvec2 = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners2, charuco_ids2, self.charucoBoardPaper, mtx, dist, None, None)
        else:
            return False
        if not retval2 or not retval1:
            return False
        tvec1 = tvec1.flatten()
        tvec2 = tvec2.flatten()
        R1 = cv2.Rodrigues(rvec1.flatten())[0]
        R2 = cv2.Rodrigues(rvec2.flatten())[0]
        # get the pose of the marker in each camera frame
        pose1 = np.eye(4)
        pose1[:3, :3] = R1
        pose1[:3, 3] = tvec1
        pose2 = np.eye(4)
        pose2[:3, :3] = R2
        pose2[:3, 3] = tvec2
        if IR:
            self.cam1.setIRPose(pose1)
            self.cam2.setIRPose(pose2)
            self.relativePoseIR = np.matmul(pose1, np.linalg.inv(pose2))
        else:
            self.cam1.setVisPose(pose1)
            self.cam2.setVisPose(pose2)
            self.relativePoseVis = np.matmul(pose1, np.linalg.inv(pose2))
        # get the relative pose of the two cameras
        return True

    def get3dPoint(self, point1: np.ndarray, point2: np.ndarray, IR=True) -> np.ndarray:
        # get the 3d point from the two 2d points
        if IR:
            undistort1 = cv2.undistortImagePoints(
                point1, self.cam1.irMtx, self.cam1.irDist)
            undistort2 = cv2.undistortImagePoints(
                point2, self.cam2.irMtx, self.cam2.irDist)
            position = cv2.triangulatePoints(
                self.cam1.irProjection, self.cam2.irProjection, undistort1, undistort2)
        else:
            undistort1 = cv2.undistortImagePoints(
                point1, self.cam1.visMtx, self.cam1.visDist)
            undistort2 = cv2.undistortImagePoints(
                point2, self.cam2.visMtx, self.cam2.visDist)
            position = cv2.triangulatePoints(
                self.cam1.visProjection, self.cam2.visProjection, undistort1, undistort2)
        position = position.reshape(4)
        return (position[:3]/position[3])
    
    def findCharucoCalibration(self, cap1, cap2, offsetx, offsety):
        # figure out coordinates of this and add white boarder
        imS = cv2.imread("src/ChArUco_Marker_Display.png")
        res = imS.shape
        imS = cv2.resize(imS, (int(res[1]/1.2), int(res[0]/1.2)))
        cv2.imshow("Aruco", imS)
        # move window to top left
        cv2.moveWindow("Aruco", offsetx, offsety)
        cv2.waitKey(1)
        time.sleep(1)
        screenShot = np.array(ImageGrab.grab())
        screenShot = cv2.cvtColor(screenShot, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("screenShot.png", screenShot)
        # screenShot = cv2.imread("screenShot.png")
        # screenShot = cv2.imread("src/ChArUco_Marker.png")
        (screenShotCorners, idsScreenShot,
         rejectedImgPoints) = self.ArucoDetector.detectMarkers(screenShot)
        screenShotCorners, idsScreenShot = self.getCharucoCorner(
            screenShotCorners, idsScreenShot, rejectedImgPoints, screenShot, None)
        idsScreenShot = idsScreenShot.flatten()
        idOrder = np.argsort(idsScreenShot)
        print(idOrder.shape)
        nCorners = 12
        screenShotCornersOut = np.zeros((nCorners, 2))
        for i, id in enumerate(idOrder):
            screenShotCornersOut[i] = screenShotCorners[id][0]
        # do a screen grab of the display
        # find the marker in the screen shot to get pixel coordinates
        cam1Done = False
        cam2Done = False
        while True:
            # read in images
            if not cam1Done:
                grabbed1, frame1 = cap1.read()
            if not cam2Done:
                grabbed2, frame2 = cap2.read()
            # frame1 = cv2.imread("cam1.png")
            # frame2 = cv2.imread("cam2.png")
            gray1, r1, g1 = cv2.split(frame1)
            gray2, r2, g2 = cv2.split(frame2)
            # gray1 = cv2.imread("gray1.png")
            # gray2 = cv2.imread("gray2.png")
            # detect markers
            if not cam1Done:
                (markerCorners1, markerIds1,
                 rejectedImgPoints1) = self.ArucoDetector.detectMarkers(gray1)
                if markerIds1 is None:
                    continue
                corners1, ids1 = self.getCharucoCorner(
                    markerCorners1, markerIds1, rejectedImgPoints1, gray1, self.cam1)
            if not cam2Done:
                (markerCorners2, markerIds2,
                 rejectedImgPoints2) = self.ArucoDetector.detectMarkers(gray2)
                if markerIds2 is None:
                    continue
                corners2, ids2 = self.getCharucoCorner(
                    markerCorners2, markerIds2, rejectedImgPoints2, gray2, self.cam2)
            if ids1 is not None and ids2 is not None:
                if len(ids1) == nCorners:
                    cam1Done = True
                if len(ids2) == nCorners:
                    cam2Done = True
                print(len(ids1), len(ids2))
            if ids1 is not None and ids2 is not None and len(ids1) == nCorners and len(ids2) == nCorners:
                ids1 = ids1.flatten()
                ids2 = ids2.flatten()
                cam1Corners = np.zeros((nCorners, 2))
                cam2Corners = np.zeros((nCorners, 2))
                idOrder = np.argsort(ids1)
                for i, id in enumerate(idOrder):
                    cam1Corners[i] = corners1[id][0]
                idOrder = np.argsort(ids2)
                for i, id in enumerate(idOrder):
                    cam2Corners[i] = corners2[id][0]
                points = np.zeros((nCorners, 3))
                for i in range(nCorners):
                    points[i] = self.get3dPoint(
                        cam1Corners[i], cam2Corners[i], False)
                cv2.destroyAllWindows()
                return screenShotCornersOut, points, cam1Corners, cam2Corners

    def getProjectorPositionStream(self, cap1: cv2.VideoCapture, cap2: cv2.VideoCapture):
        nCorners = 12
        screenShotCorners = np.zeros((nCorners, 2))
        points = np.zeros((nCorners, 3))
        for i in range(10):
            screenShotCornersOut, newPoints, c1Corners, c2Corners = self.findCharucoCalibration(cap1, cap2, 10*i, 10*i)
            screenShotCorners[nCorners*i:nCorners*(i+1)] = screenShotCornersOut
            points[nCorners*i:nCorners*(i+1)] = newPoints
            if i ==0:
                cam1Corners = c1Corners
                cam2Corners = c2Corners
        xhat, yhat, zhat, offSet = getPlaneVectors(points)
        pose1 = getTransformationMatrix(xhat, yhat, zhat, offSet)
        pose2 = np.matmul(np.linalg.inv(self.relativePoseVis), pose1)
        self.cam1.setVisPose(pose1)
        self.cam2.setVisPose(pose2)
        self.cam1.setIRPose(np.matmul(np.linalg.inv(self.cam1VisToIRPose), pose1))
        self.cam2.setIRPose(
            np.matmul(np.linalg.inv(self.relativePoseIR), self.cam1.irPose))
        TL = self.get3dPoint(cam1Corners[10], cam2Corners[10], False)
        TR = self.get3dPoint(
            cam1Corners[0], cam2Corners[0], False)
        BL = self.get3dPoint(cam1Corners[11], cam2Corners[11], False)
        BR = self.get3dPoint(
            cam1Corners[1], cam2Corners[1], False)
        print(TL, TR, BL, BR)
        screenTL = screenShotCornersOut[10]
        screenTR = screenShotCornersOut[0]
        screenBL = screenShotCornersOut[11]
        screenBR = screenShotCornersOut[1]
        points1 = np.array(
            [TL[:2], TR[:2], BL[:2], BR[:2]], dtype='float32')
        points2 = np.array(
            [screenTL, screenTR, screenBL, screenBR], dtype='float32')
        matrix = cv2.getPerspectiveTransform(points1, points2)
        return matrix


def getTransformationMatrix(x, y, z, offset):
    T = np.eye(4)
    # might need to subtract TL from the 3 rows one post said so, but cant tell till we test and fix the calibration
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = offset
    return T


def getPlaneVectors(points):
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    result = opt.minimize(planeErr, [0, 0, 0], args=(
        xs, ys, zs), method='Nelder-Mead', tol=1e-6)
    mse = planeErr(result.x, xs, ys, zs)
    print(mse, np.sqrt(mse))
    a, b, c = result.x
    v1 = np.array([xs[10], ys[10], planeZ(xs[10], ys[10], a, b, c)])
    v2 = np.array([xs[0], ys[0], planeZ(xs[0], ys[0], a, b, c)])
    xhat = v2 - v1
    zhat = np.array([a, b, 1])
    xhat = xhat/np.linalg.norm(xhat)
    zhat = zhat/np.linalg.norm(zhat)
    yhat = np.cross(zhat, xhat)
    return xhat, yhat, zhat, v1


def planeErr(coefs, x, y, z):
    return np.mean(np.square(1000*(coefs[0]*x + coefs[1]*y + z - coefs[2])))


def planeZ(x, y, a, b, c):
    return (c - a*x - b*y)
