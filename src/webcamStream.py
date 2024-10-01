import cv2


def openStream(src: int = 0, height: int = 1080, width: int = 1920, fps: int = 30, focus: int = 0, exposure: int = -7) -> cv2.VideoCapture:
    stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    stream.open(src, cv2.CAP_DSHOW)
    stream.set(cv2.CAP_PROP_FPS, fps)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    stream.set(cv2.CAP_PROP_FOURCC,
               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # lower focus focuses further away from the camera
    # focus min: 0, max: 255, increment:5
    stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    stream.set(cv2.CAP_PROP_EXPOSURE, exposure)
    stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    stream.set(cv2.CAP_PROP_FOCUS, focus)
    return stream
