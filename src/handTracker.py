from threading import Thread
import cv2
import time
import mediapipe as mp
import numpy as np
mp_hands = mp.solutions.hands


class HandTracker:
    def __init__(self, stream: cv2.VideoCapture):
        # initialize the camera and properties
        self.stream = stream
        # self.stream.set(cv2.CAP_PROP_FPS, fps)
        (self.grabbed, self.frame) = self.stream.read()
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        # self.stream.set(cv2.CAP_PROP_FPS, fps)
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopCap = False
        self.stopTrack = False
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.processedFrame = self.frame
        self.hand_landmarks = None
        self.drawDebug = True

    def startCapture(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name="capture", args=())
        t.daemon = True
        t.start()
        return

    def startHandTracking(self):
        t = Thread(target=self.handThread, name="HandTracker", args=())
        t.daemon = True
        t.start()
        return

    def update(self):
        frameDelta = 1/self.fps
        # keep looping infinitely until the thread is stopped
        # clear the camera buffer on boot
        for i in range(10):
            self.stream.grab()
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopCap:
                return

            prevTime = time.time()
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            sleepTime = frameDelta - (time.time() - prevTime) - 0.01
            time.sleep(sleepTime*(sleepTime > 0))

    def detectHands(self):
        frame = self.frame.copy()
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

    def handThread(self):
        delta = 1/self.fps
        while True:
            if self.stopTrack:
                return
            self.detectHands()

    def read(self):
        # return the frame most recently read
        return self.frame

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
