import os
import numpy as np
import cv2

# ------------------------------
# ENTER YOUR PARAMETERS HERE:
ARUCO_DICT = cv2.aruco.DICT_4X4_1000
SQUARES_VERTICALLY = 4
SQUARES_HORIZONTALLY = 6
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
LENGTH_PX = 3400   # total length of the page in pixels
MARGIN_PX = 100   # size of the margin in pixels
SAVE_NAME = 'ChArUco_Marker.png'
# ------------------------------


def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(
        board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    cv2.imshow("img", img)
    cv2.waitKey(2000)
    cv2.imwrite(SAVE_NAME, img)


create_and_save_new_board()
