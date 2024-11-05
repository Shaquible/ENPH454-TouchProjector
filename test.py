import cv2
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
ArucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)
im = cv2.imread("src/arucoQuad.png")
corners, ids, rej = ArucoDetector.detectMarkers(im)
for corner, id in zip(corners, ids):
    print(corner, id)
