import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import glob

def adjustChannelVals(im, channelScales):
    h,s,v = cv2.split(im)
    h = cv2.multiply(h, channelScales[0])
    s = cv2.multiply(s, channelScales[1])
    v = cv2.multiply(v, channelScales[2])
    h = cv2.add(h, channelScales[3])
    s = cv2.add(s, channelScales[4])
    v = cv2.add(v, channelScales[5])
    return cv2.merge((h,s,v))

def subtractImageOpt(channelScales, base, dist, target):
    mask = cv2.inRange(dist, np.array([1,1,1]), np.array([255,255,255]))
    dist = adjustChannelVals(dist, channelScales)
    
    h,s,v = cv2.split(base)
    h = cv2.bitwise_and(h, mask)
    s = cv2.bitwise_and(s, mask)
    v = cv2.bitwise_and(v, mask)
    base = cv2.merge((h,s,v))
    h,s,v = cv2.split(target)
    h = cv2.bitwise_and(h, mask)
    s = cv2.bitwise_and(s, mask)
    v = cv2.bitwise_and(v, mask)
    target = cv2.merge((h,s,v))
    sub = cv2.subtract(base, dist)
    diff = cv2.absdiff(sub, target)
    return np.sum(diff)

def processImage(im, reference, target, params, matrix, col, row):
    dst = cv2.warpPerspective(reference,matrix,(col,row))
    dst = adjustChannelVals(dst, params.x)
    sub = cv2.subtract(im, dst)
    DiffToTarget = cv2.absdiff(sub, target)
    h,s,v = cv2.split(DiffToTarget)
    mask = cv2.inRange(s, 35, 100)
    h= np.ones_like(h)*10
    #shift saturation to 0
    s= np.ones_like(h)*150
    h = cv2.bitwise_and(h, mask)
    s = cv2.bitwise_and(s, mask)
    v = np.ones_like(h)*255
    v = cv2.bitwise_and(v, mask)
    return cv2.merge((h,s,v))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
ArucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)

#aruco test
im = cv2.imread('displayedAruco.jpg')
grayScale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
(corners1, ids1, rejectedImgPoints) = ArucoDetector.detectMarkers(im)
corners1 = corners1[0][0]
im = cv2.imread("aruco.PNG")
grayScale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
(corners2, ids2, rejectedImgPoints) = ArucoDetector.detectMarkers(im)
corners2 = corners2[0][0]
#change order so 2 and 3 are swapped
temp = corners2[2].copy()
corners2[2] = corners2[3]
corners2[3] = temp
temp = corners1[2].copy()
corners1[2] = corners1[3]
corners1[3] = temp
print(corners1)
print(corners2)
#correct order [[0,0],[352,0],[0,352],[352,352]]

#optimizing overlay
camPicture = cv2.imread("colorTestDisp.jpg")
camPicture = cv2.cvtColor(camPicture, cv2.COLOR_BGR2HSV)
row, col, ch = camPicture.shape
print(camPicture.shape)
refImage = cv2.imread("colorTestRef.PNG")
refImage = cv2.cvtColor(refImage, cv2.COLOR_BGR2HSV)
targetImage = cv2.imread("emptyProj.jpg")
targetImage = cv2.cvtColor(targetImage, cv2.COLOR_BGR2HSV)
#resize to 1920x1080
print(refImage.shape)
pts1 = np.float32(corners1)
pts2 = np.float32(corners2)
matrix = cv2.getPerspectiveTransform(pts2,pts1)
dst = cv2.warpPerspective(refImage,matrix,(col,row))

params = minimize(subtractImageOpt, [1,1,1,0,0,0], args=(camPicture, dst, targetImage), method='Nelder-Mead', bounds=((-10, 10), (-10, 10), (-10, 10), (-255, 255), (-255, 255), (-255, 255)))
print(params.x)

folders = glob.glob("Slide*")
for folder in folders:
    refImage = glob.glob(folder + "/Slide*.PNG")[0]
    refImage = cv2.imread(refImage)
    refImage = cv2.cvtColor(refImage, cv2.COLOR_BGR2HSV)
    images = glob.glob(folder + "/*.jpg")
    for image in images:
        if "Out" in image:
            continue
        im = cv2.imread(image)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        out = processImage(im, refImage, targetImage, params, matrix, col, row)
        out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
        cv2.imwrite(image.split(".")[0] + "Out.jpg", out)