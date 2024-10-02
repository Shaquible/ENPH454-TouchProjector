from handTracker import HandTracker
from webcamStream import openStream
from triangulation import Triangulation, Camera
import mediapipe as mp
import numpy as np
from multiprocessing import Process, Queue


def ML_Process(Tracker: HandTracker, dataQueue: Queue, killQueue: Queue):
    Tracker.startCapture()
    Tracker.startHandTracking(dataQueue)
    Tracker.watchKill(killQueue)


def main():
    imHeight = 1080
    imWidth = 1920
    mp_hands = mp.solutions.hands
    markerWidth = 0.1586
    npfile = np.load("cameraIntrinsics/cam128-2.npz")
    mtx1 = npfile["mtx"]
    dist1 = npfile["dist"]
    npfile = np.load("cameraIntrinsics/cam127-2.npz")
    mtx2 = npfile["mtx"]
    dist2 = npfile["dist"]
    cap1 = openStream(0, imHeight, imWidth, exposure=-8)
    cap2 = openStream(1, imHeight, imWidth, exposure=-8)

    tri = Triangulation(Camera(mtx1, dist1), Camera(mtx2, dist2))
    tri.getCameraPositionsStream(cap1, cap2, markerWidth)
    print("Position Found")
    tracker1 = HandTracker(cap1)
    tracker1.drawDebug = False
    tracker2 = HandTracker(cap2)
    tracker2.drawDebug = False
    # launching the process to run tracking on each camera
    procs = []
    q1 = Queue(1)
    q2 = Queue(1)
    kill1 = Queue(1)
    kill2 = Queue(1)
    procs.append(Process(target=ML_Process, args=(tracker1, q1, kill1)))
    procs.append(Process(target=ML_Process, args=(tracker2, q2, kill2)))
    for proc in procs:
        proc.start()

    try:
        while True:
            cam1Hands = q1.get()
            cam2Hands = q2.get()
            if cam1Hands is not None and cam2Hands is not None:
                for hand in cam1Hands:
                    cam1Coords = np.array([hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*imWidth,
                                           hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*imHeight])
                for hand in cam2Hands:
                    cam2Coords = np.array([hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*imWidth,
                                           hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*imHeight])
                pos = tri.get3dPoint(cam1Coords, cam2Coords)[:, 0]
                print(pos, end="\r")
    except KeyboardInterrupt:
        kill1.put(1)
        kill2.put(1)
        for proc in procs:
            proc.join()
        return


if __name__ == "__main__":
    main()
