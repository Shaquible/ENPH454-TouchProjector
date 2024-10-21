from handTracker import HandTracker
from webcamStream import openStream
from triangulation import Triangulation, Camera
import mediapipe as mp
import numpy as np
from multiprocessing import Process, Queue
import time
from mouseMove import mouseMove
from handPositionFilter import rollingAvg
imHeight = 1080
imWidth = 1920
exposure = -8
def ML_Process(captureNum: int, dataQueue: Queue, capSQ: Queue, capRQ: Queue, processSQ: Queue, processRQ: Queue):
    open = False
    while open is False:
        cap = openStream(captureNum, imHeight, imWidth, exposure=exposure)
        open = cap.isOpened()
    time.sleep(0.1)
    Tracker = HandTracker(cap)
    Tracker.drawDebug = False
    Tracker.startCapture(capSQ, capRQ)
    Tracker.startHandTracking(dataQueue, processSQ, processRQ)


def main():
   
    mp_hands = mp.solutions.hands
    markerWidth = 0.1586
    npfile = np.load("cameraIntrinsics/cam128-2.npz")
    mtx1 = npfile["mtx"]
    dist1 = npfile["dist"]
    npfile = np.load("cameraIntrinsics/cam127-2.npz")
    mtx2 = npfile["mtx"]
    dist2 = npfile["dist"]
    cap1 = openStream(0, imHeight, imWidth, exposure=exposure)
    cap2 = openStream(1, imHeight, imWidth, exposure=exposure)

    tri = Triangulation(Camera(mtx1, dist1), Camera(mtx2, dist2))
    tri.getCameraPositionsStream(cap1, cap2, markerWidth)
    print("Position Found")
    cap1.release()
    cap2.release()
    time.sleep(0.5)
    handAverage = rollingAvg(2,3,3)
    # launching the process to run tracking on each camera
    procs = []
    q1 = Queue(1)
    q2 = Queue(1)
    capQ1 = Queue(1)
    capQ2 = Queue(1)
    processQ1 = Queue(1)
    processQ2 = Queue(1)
    procs.append(Process(target=ML_Process, args=(0, q1, capQ1, capQ2, processQ1, processQ2)))
    procs.append(Process(target=ML_Process, args=(1, q2, capQ2, capQ1, processQ2, processQ1)))
    for proc in procs:
        proc.start()
    mouse = mouseMove()
    try:
        while True:
            try:
                cam1Hands = q1.get(timeout=0.01)
                cam2Hands = q2.get(timeout=0.01)
            except:
                continue
            if cam1Hands is not None and cam2Hands is not None:
                for hand in cam1Hands:
                    cam1Coords = np.array([hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*imWidth,
                                           hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*imHeight])
                for hand in cam2Hands:
                    cam2Coords = np.array([hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*imWidth,
                                           hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*imHeight])
                pos = tri.get3dPoint(cam1Coords, cam2Coords)[:, 0]
                pos = handAverage.smoothPos(pos)
                position = "X: {:.2f} Y: {:.2f} Z: {:.2f}".format(pos[0]*100, pos[1]*100, pos[2]*100)
                print(position, end="\r")
                mouse.moveMouse(pos)

                #print(pos, end="\r")
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.join()
        return


if __name__ == "__main__":
    main()
