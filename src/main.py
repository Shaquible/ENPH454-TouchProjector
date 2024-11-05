from handTracker import HandTracker
from webcamStream import openStream
from triangulation import Triangulation, Camera
import mediapipe as mp
import numpy as np
from multiprocessing import Process, Queue
import time
from mouseMove import mouseMove
from handPositionFilter import deEmphasis
import cv2
import pandas as pd
imHeight = 1080
imWidth = 1920
exposure = -8


def ML_Process(captureNum: int, cropRegion, dataQueue: Queue, capSQ: Queue, capRQ: Queue, processSQ: Queue, processRQ: Queue):
    opened = False
    while not opened:
        cap = openStream(captureNum, 1080, 1920, exposure=-8)
        if cap.isOpened():
            opened = True

    Tracker = HandTracker(cap, captureNum, cropRegion)
    # Tracker.drawDebug = False
    Tracker.startCapture(capSQ, capRQ)
    time.sleep(0.2)
    Tracker.startHandTracking(dataQueue, processSQ, processRQ)
    Tracker.watchKill()


def main():

    mp_hands = mp.solutions.hands
    markerWidth = 0.1586
    cap1 = openStream(0, imHeight, imWidth, exposure=exposure)
    cap2 = openStream(1, imHeight, imWidth, exposure=exposure)
    npfile = np.load("cameraIntrinsics/IRCam1Visible.npz")
    mtx1 = npfile["mtx"]
    dist1 = npfile["dist"]
    npfile = np.load("cameraIntrinsics/IRCam2Visible.npz")
    mtx2 = npfile["mtx"]
    dist2 = npfile["dist"]
    npfile = np.load("src/relativePose.npy")
    # need to crop the images
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 and ret2:
            break
    crop1 = cv2.selectROI("cam0", frame1)
    cv2.destroyAllWindows()
    crop2 = cv2.selectROI("cam1", frame2)
    cv2.destroyAllWindows()

    tri = Triangulation(Camera(mtx1, dist1), Camera(mtx2, dist2))
    tri.relativePose = npfile
    tri.cam2.setPose(np.linalg.inv(tri.relativePose))
    xy_to_uv_mat, camFlip = tri.getProjectorPositionStream(cap1, cap2)
    pose1 = tri.cam1.pose
    pose2 = tri.cam2.pose
    npfile = np.load("cameraIntrinsics/IRCam1.npz")
    mtx1 = npfile["mtx"]
    dist1 = npfile["dist"]
    npfile = np.load("cameraIntrinsics/IRCam2.npz")
    mtx2 = npfile["mtx"]
    dist2 = npfile["dist"]
    # check if in the orientation the camera order was flipped
    if camFlip:
        tri = Triangulation(Camera(mtx2, dist2), Camera(mtx1, dist1))
    else:
        tri = Triangulation(Camera(mtx1, dist1), Camera(mtx2, dist2))
    tri.relativePose = npfile
    tri.cam1.setPose(pose1)
    tri.cam2.setPose(pose2)
    # tri.getCameraPositionsStream(cap1, cap2, markerWidth)
    print("Position Found")
    cap1.release()
    cap2.release()
    time.sleep(0.2)
    positionFilter = deEmphasis()
    # launching the process to run tracking on each camera
    procs = []
    q1 = Queue(1)
    q2 = Queue(1)
    capQ1 = Queue(1)
    capQ2 = Queue(1)
    processQ1 = Queue(1)
    processQ2 = Queue(1)
    procs.append(Process(target=ML_Process, args=(
        0, crop1, q1, capQ1, capQ2, processQ1, processQ2)))
    procs.append(Process(target=ML_Process, args=(
        1, crop2, q2, capQ2, capQ1, processQ2, processQ1)))
    for proc in procs:
        proc.start()
    mouse = mouseMove(xy_to_uv_mat)
    dataCollectLen = 500
    times = np.zeros(dataCollectLen)
    xs = np.zeros(dataCollectLen)
    ys = np.zeros(dataCollectLen)
    zs = np.zeros(dataCollectLen)
    i = 0
    t0 = time.time()
    try:
        while True:
            cam1Hands = q1.get()
            cam2Hands = q2.get()
            if cam1Hands is not None and cam2Hands is not None:
                for hand in cam1Hands:
                    cam1Coords = np.array([(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)*crop1[2] + crop1[0],
                                           (hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)*crop1[3] + crop1[1]])
                for hand in cam2Hands:
                    cam2Coords = np.array([(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)*crop2[2] + crop2[0],
                                           (hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)*crop2[3] + crop2[1]])
                pos = tri.get3dPoint(cam1Coords, cam2Coords)
                # if i < dataCollectLen:
                #     times[i] = time.time()
                #     xs[i] = pos[0]
                #     ys[i] = pos[1]
                #     zs[i] = pos[2]
                #     i += 1
                # if i == dataCollectLen:
                #     df = pd.DataFrame({"Time": times, "X": xs, "Y": ys, "Z": zs})
                #     df.to_csv("dataCapture.csv")
                #     i += 1
                #     print("END\n")

                dt = time.time() - t0
                t0 = time.time()

                # pos = positionFilter.smoothPos(pos)
                position = "X: {:.2f} Y: {:.2f} Z: {:.2f} dt{:.3f}".format(
                    pos[0]*100, pos[1]*100, pos[2]*100, dt)
                print(position, end="\r")
                mouse.moveMouse(pos)

                # print(pos, end="\r")

    except KeyboardInterrupt:
        print("\nShutting down")
        for proc in procs:
            proc.join()
        return


if __name__ == "__main__":
    main()
