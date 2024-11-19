from handTracker import HandTracker
from webcamStream import openStream
from triangulationCharuco import Triangulation, Camera
import mediapipe as mp
import numpy as np
from multiprocessing import Process, Queue
import time
from mouseMove import mouseMove
from handPositionFilter import deEmphasis
import cv2
from picoControl import PicoControl
import pandas as pd
import Gesture
import autoCrop

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
    pico = PicoControl("COM3", 0)
    pico.setIRCutFilter(1)
    time.sleep(1)
    cap1 = openStream(0, imHeight, imWidth, exposure=exposure)
    cap2 = openStream(1, imHeight, imWidth, exposure=exposure)
    # loading intrinsics and poses
    npfile = np.load("cameraIntrinsics/Cam1Vis.npz")
    mtx1 = npfile["mtx"]
    dist1 = npfile["dist"]
    npfile = np.load("cameraIntrinsics/Cam2Vis.npz")
    mtx2 = npfile["mtx"]
    dist2 = npfile["dist"]
    npfile = np.load("cameraIntrinsics/Cam1IR.npz")
    mtx1IR = npfile["mtx"]
    dist1IR = npfile["dist"]
    npfile = np.load("cameraIntrinsics/Cam2IR.npz")
    mtx2IR = npfile["mtx"]
    dist2IR = npfile["dist"]

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

    tri = Triangulation(Camera(mtx1IR, dist1IR, mtx1, dist1),
                        Camera(mtx2IR, dist2IR, mtx2, dist2))
    npfile = np.load("cameraIntrinsics/relativePoses.npz")
    tri.relativePoseVis = npfile["relativePoseVis"]
    tri.relativePoseIR = npfile["relativePoseIR"]
    tri.cam1VisToIRPose = npfile["cam1VisToIRPose"]
    tri.cam2.setVisPose(np.linalg.inv(tri.relativePoseVis))

    # converts real space to pixel space in the projector
    xy_to_uv_mat = tri.getProjectorPositionStream(cap1, cap2)
    # Crops the image to the area around the projector screen in each camera stream.
    print(xy_to_uv_mat)
    #crop1, crop2 = autoCrop.crop(xy_to_uv_mat, tri.cam1, tri.cam2, 0.1)
    # tri.getCameraPositionsStream(cap1, cap2, markerWidth)
    print("Position Found")
    cap1.release()
    cap2.release()
    time.sleep(0.2)
    pico.setIRCutFilter(0)
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
    # initializes the gesture recognition class that can quickly fetch gestures from images.
    gesture = Gesture.Gesture(imWidth, imHeight)
    dataCollectLen = 2000
    times = np.zeros(dataCollectLen)
    xs = np.zeros(dataCollectLen)
    ys = np.zeros(dataCollectLen)
    zs = np.zeros(dataCollectLen)
    i = 0
    t0 = time.time()
    # main loop
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
                hand_sign_id = gesture.getGesture(hand.landmark)
                #Shand_sign_id = 0

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

                pos = positionFilter.smoothPos(pos)
                position = "X: {:.2f} Y: {:.2f} Z: {:.2f} dt{:.3f} ID {}         ".format(
                    pos[0]*100, pos[1]*100, pos[2]*100, dt, hand_sign_id)
                print(position, end="\r")
                mouse.moveMouse(pos)
            # else:
            #     mouse.debounceZ.buffer= np.zeros(mouse.debounceZ.N, dtype = bool)
            #     mouse.lastState = False
            #     mouse.unclick()

                # print(pos, end="\r")

    except KeyboardInterrupt:
        print("\nShutting down")
        for proc in procs:
            proc.join()
        return


if __name__ == "__main__":
    main()
