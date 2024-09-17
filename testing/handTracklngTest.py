import cv2
import mediapipe as mp
from webcamStream import WebcamVideoStream
import time
mp_hands = mp.solutions.hands

# Initialize the webcam stream
webcam = WebcamVideoStream()
webcam.start()
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    prevTime = 0
    while True:
        frame = webcam.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # draw the hand landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        dt = time.time() - prevTime
        prevTime = time.time()
        cv2.putText(frame, "FPS: " + format(1/dt, '.1f'), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            webcam.stop()
            cv2.destroyAllWindows()
            break
