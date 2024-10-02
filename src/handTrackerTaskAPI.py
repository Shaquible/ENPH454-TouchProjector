from threading import Thread
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp


def draw_landmarks_on_image(annotated_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
    return annotated_image


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
        base_options = python.BaseOptions(
            model_asset_path='models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
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
        start = time.time()
        frame = self.frame.copy()
        mp_im = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.detector.detect(mp_im)
        if len(results.hand_landmarks) != 0:
            if self.drawDebug:
                frame = draw_landmarks_on_image(
                    frame, results)
                self.processedFrame = frame
            self.hand_landmarks = results.hand_landmarks

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
        self.detector.close()


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
