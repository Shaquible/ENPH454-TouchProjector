import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

class Gesture:
    # Initializes the gesture recognition protocol including the image dimensions which will be used later.
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    def getGesture(self, hand_landmarks):
        
        #Import the classification of gestures from the trained models in KeyPointClassifier
        keypoint_classifier = KeyPointClassifier()
        
        hand_sign_id = 0
        
        if hand_landmarks is not None:
            # Landmark calculation
            landmark_list = calc_landmark_list(self.image_width, self.image_height, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)

            # Hand sign classification (This is the part that is needed to recognize if the finger is out or)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            
            # Hand Sign ID is 2 for a pointed finger
                   
        return hand_sign_id

#This creates a list of landmarks relative to the screensize from the given landmarks.
def calc_landmark_list(image_width, image_height, landmarks):

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point  

# This processes the list of landmarks to be ready to be read by the upcoming
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list