import cv2
import time
src = cv2.VideoCapture(0)
src.open(0, cv2.CAP_DSHOW)
src.set(cv2.CAP_PROP_AUTOFOCUS, 0)
time.sleep(2)
for i in range(51):
    src.set(cv2.CAP_PROP_FOCUS, i*5)
    time.sleep(0.5)
    ret, frame = src.read()
    cv2.imwrite("frame{}.jpg".format(i), frame)