from threading import Thread
import cv2
import time
import mediapipe as mp
import numpy as np
from multiprocessing import Queue
mp_hands = mp.solutions.hands


class HandTracker:
    def __init__(self, stream: cv2.VideoCapture, capNum: int, cropRegion, fps: int = 30, height=1080, width=1920):
        # initialize the camera and properties
        self.stream = stream
        # self.stream.set(cv2.CAP_PROP_FPS, fps)
        (self.grabbed, self.frame) = self.stream.read()
        while not self.grabbed:
            (self.grabbed, self.frame) = self.stream.read()
        self.H = np.ones((cropRegion[3], cropRegion[2]), dtype=np.uint8)*15
        self.S = np.ones((cropRegion[3], cropRegion[2]), dtype=np.uint8)*150
        self.fps = fps
        self.cropRegion = cropRegion
        # self.stream.set(cv2.CAP_PROP_FPS, fps)
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopCap = False
        self.stopTrack = False
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.3, model_complexity=1)
        self.processedFrame = self.frame
        self.hand_landmarks = None
        self.drawDebug = False
        self.num = capNum

    def startCapture(self, sendQueue: Queue, receiveQueue: Queue):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name="capture",
                   args=(sendQueue, receiveQueue))
        t.daemon = True
        t.start()
        return

    def startHandTracking(self, dataQueue: Queue, sendQueue: Queue, receiveQueue: Queue):
        t = Thread(target=self.handThread,
                   name="HandTracker", args=(dataQueue, sendQueue, receiveQueue))
        t.daemon = True
        t.start()
        return

    def update(self, sendQueue: Queue, receiveQueue: Queue):
        frameDelta = 1/self.fps
        # keep looping infinitely until the thread is stopped
        # clear the camera buffer on boot
        for i in range(10):
            self.stream.grab()
        while True:
            sendQueue.put(1)
            receiveQueue.get()
            # if the thread indicator variable is set, stop the thread
            if self.stopCap:
                return

            prevTime = time.perf_counter()
            # otherwise, read the next frame from the stream
            (self.grabbed, frame) = self.stream.read()
            while not self.grabbed:
                (self.grabbed, frame) = self.stream.read()
            self.frame = frame
            sleepTime = frameDelta - (time.perf_counter() - prevTime) - 0.01
            time.sleep(sleepTime*(sleepTime > 0))

    def detectHands(self):
        if self.frame is None:
            return
        frame = self.frame.copy()
        # crop the image
        frame = frame[int(self.cropRegion[1]):int(self.cropRegion[1]+self.cropRegion[3]),
                      int(self.cropRegion[0]):int(self.cropRegion[0]+self.cropRegion[2])]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        merge = cv2.merge((self.H, self.S, v))
        frame = cv2.cvtColor(merge, cv2.COLOR_HSV2BGR)
        results = self.hands.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            if self.drawDebug:
                for hand_landmarks in results.multi_hand_landmarks:

                    # draw the hand landmarks on the frame
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                self.processedFrame = frame
            self.hand_landmarks = results.multi_hand_landmarks
        else:
            self.hand_landmarks = None

    def handThread(self, dataQueue: Queue, sendQueue: Queue, receiveQueue: Queue):
        minTime = 1/self.fps
        while True:
            startTime = time.perf_counter()
            if self.stopTrack:
                return
            sendQueue.put(1)
            receiveQueue.get()
            self.detectHands()
            if not dataQueue.empty():
                try:
                    dataQueue.get(timeout=0.01)
                except:
                    pass
            dataQueue.put(self.hand_landmarks)
            dT = time.perf_counter() - startTime
            if dT < minTime:
                time.sleep(minTime - dT)
            # cv2.imshow(str(self.num), self.processedFrame)
            # cv2.waitKey(1)

    def read(self):
        # return the frame most recently read
        return self.frame

    def watchKill(self):
        try:
            while True:
                time.sleep(1)
        except:
            self.shutdown()
            return

    def stopCapture(self):
        # indicate that the thread should be stopped
        self.stopCap = True
        self.stream.release()

    def stopTracking(self):
        self.stopTrack = True

    def shutdown(self):
        self.stopCapture()
        self.stopTracking()
        self.hands.close()


if __name__ == "__main__":
    import webcamStream
    stream = webcamStream.openStream(exposure=-3)
    webcam = HandTracker(stream)
    webcam.startCapture()
    webcam.startHandTracking()
    try:
        while True:
            if webcam.frame is not None:
                cv2.imshow("Hand Tracking", webcam.processedFrame)
                if cv2.waitKey(3) & 0xFF == ord('q'):
                    webcam.shutdown()
                    cv2.destroyAllWindows()
                    break
    except KeyboardInterrupt:
        webcam.shutdown()
        cv2.destroyAllWindows()
