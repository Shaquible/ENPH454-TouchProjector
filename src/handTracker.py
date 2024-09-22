from threading import Thread
import cv2
import time
import mediapipe as mp
mp_hands = mp.solutions.hands


class HandTracker:
    def __init__(self, src=0, name="WebcamVideoStream", height=1080, width=1920, fps=30, focus=0):
        # initialize the camera and properties
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.fps = fps
        self.stream.open(src, cv2.CAP_DSHOW)

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # lower focus focuses further away from the camera
        # focus min: 0, max: 255, increment:5
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.stream.set(cv2.CAP_PROP_FOCUS, focus)
        # self.stream.set(cv2.CAP_PROP_FPS, fps)
        (self.grabbed, self.frame) = self.stream.read()
        # self.stream.set(cv2.CAP_PROP_FPS, fps)
        # initialize the thread name
        self.name = name
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.processedFrame = self.frame

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        t2 = Thread(target=self.handThread, name="HandTracker", args=())
        t2.daemon = True
        t2.start()
        return self

    def update(self):
        frameDelta = 1/self.fps
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            prevTime = time.time()
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            sleepTime = frameDelta - (time.time() - prevTime)
            time.sleep(sleepTime*(sleepTime > 0))

    def detectHands(self):
        frame = self.frame.copy()
        results = self.hands.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # draw the hand landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        self.processedFrame = frame

    def handThread(self):
        delta = 1/self.fps
        while True:
            if self.stopped:
                return
            startTime = time.time()
            self.detectHands()
            timeLeft = delta - (time.time() - startTime)
            print(1/(delta-timeLeft), end="\r")

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()
        self.hands.close()


if __name__ == "__main__":
    webcam = HandTracker(height=720, width=1280)
    webcam.start()
    try:
        while True:
            if webcam.frame is not None:
                cv2.imshow("Hand Tracking", webcam.processedFrame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    webcam.stop()
                    cv2.destroyAllWindows()
                    break
    except KeyboardInterrupt:
        webcam.stop()
        cv2.destroyAllWindows()
