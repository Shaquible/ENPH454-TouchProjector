import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def hueCorrect(image, hue=15, saturation=150):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    h= np.ones_like(h)*hue
    #shift saturation to 0
    s= np.ones_like(h)*saturation
    combined = cv2.merge([h,s,v])
    combined = cv2.cvtColor(combined, cv2.COLOR_HSV2BGR)
    return combined

