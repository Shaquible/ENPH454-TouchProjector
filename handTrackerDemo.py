from threading import Thread
import cv2
import time
import mediapipe as mp
import numpy as np
from multiprocessing import Queue
mp_hands = mp.solutions.hands


class HandTracker:
    def __init__(self):
        # initialize the camera and properties
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.3, model_complexity=1)

    def detectHands(self, frame):
        # crop the image
        results = self.hands.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results


if __name__ == "__main__":
    frame = cv2.imread("hand.jpg")
    # convert to BNW
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
   
    tracker = HandTracker()
    times = np.zeros(20)
    nPixels = np.zeros(20)
    for cropNum in range(20):
        minRow = 20*cropNum
        minCol = 35*cropNum
        maxRow = 1080-20*cropNum
        maxCol = 1920-35*cropNum
        H = np.ones((maxRow - minRow, maxCol - minCol), dtype=np.uint8)*15
        S = np.ones((maxRow - minRow, maxCol - minCol), dtype=np.uint8)*150
        frameCropped = frame[minRow:maxRow, minCol:maxCol]
        hsv = cv2.cvtColor(frameCropped, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        frameCropped = cv2.merge((H, S, v))
        frameCropped = cv2.cvtColor(frameCropped, cv2.COLOR_HSV2BGR)
        
        startTime = time.perf_counter()
        for i in range(200):
            
            results = tracker.detectHands(frameCropped)
        endTime = time.perf_counter()
        frameTime = (endTime-startTime)/200
        times[cropNum] = frameTime

        nPixels[cropNum] = frameCropped.shape[0]*frameCropped.shape[1]

    # write the results to a file
    with open("handTrackerResults.csv", "w") as f:
        for i in range(20):
            f.write("{},{}\n".format(nPixels[i], times[i]))
