from handTracker import HandTracker
from triangulation import Camera, Triangulation
from webcamStream import openStream
import cv2
import numpy as np
import time
import mediapipe as mp
from mouseMove import mouseMove

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
ret, frame = cap1.read()

tri = Triangulation(Camera(mtx1, dist1), Camera(mtx2, dist2))
tri.getCameraPositionsStream(cap1, cap2, markerWidth)
print("Position Found")
tracker1 = HandTracker(cap1)
tracker1.startCapture()
tracker2 = HandTracker(cap2)
tracker2.startCapture()
tracker1.startHandTracking()
tracker2.startHandTracking()
mouss = mouseMove()
while True:
    cam1Hands = tracker1.hand_landmarks
    cam2Hands = tracker2.hand_landmarks
    if cam1Hands is not None and cam2Hands is not None:
        frame = tracker1.processedFrame
        for hand in cam1Hands:
            cam1Coords = np.array([hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*imWidth,
                                   hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*imHeight])
        for hand in cam2Hands:
            cam2Coords = np.array([hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*imWidth,
                                   hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*imHeight])
        position = tri.get3dPoint(cam1Coords, cam2Coords)[:,0]
        #mouss.moveMouse(position)
        cv2.putText(frame, f"X: {position[0]:.4f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Y: {position[1]:.4f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Z: {position[2]:.4f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("3D Position", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tracker1.shutdown()
            tracker2.shutdown()
            break
