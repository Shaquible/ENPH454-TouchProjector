from threading import Thread
import cv2
import time


class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream", height=1080, width=1920, fps=60, focus=0):
        # initialize the camera and properties
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.fps = fps
        self.stream.open(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # lower focus focuses further away from the camera
        # focus min: 0, max: 255, increment:5
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.stream.set(cv2.CAP_PROP_FOCUS, focus)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        (self.grabbed, self.frame) = self.stream.read()
        # self.stream.set(cv2.CAP_PROP_FPS, fps)
        # initialize the thread name
        self.name = name
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
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

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
