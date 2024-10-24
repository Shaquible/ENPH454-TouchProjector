"""
Code Authored by Keegan Kelly
"""
import cv2
import time
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#set exposure to manual
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
# lower exposure to minimum
# cap.set(cv2.CAP_PROP_EXPOSURE, -10)
# lower focus focuses further away from the camera
focus = 0  # min: 0, max: 255, increment:5
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, focus)
cap.set(cv2.CAP_PROP_FPS, 30)
i = 0
while 1:
    ret, frame = cap.read()
    if ret:

        cv2.imshow("frame", frame)
        cv2.imwrite("calibrationPhotos/chessPic"+str(i)+".png", frame)
        time.sleep(0.5)
        i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break