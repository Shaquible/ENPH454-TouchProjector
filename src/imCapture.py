# take a picture when you click space
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cap.open(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# lower focus focuses further away from the camera
# focus min: 0, max: 255, increment:5
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -8)

i = 0
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite("frame{}.jpg".format(10+i), frame)
        i += 1
