import pyautogui
from screeninfo import get_monitors
import numpy as np


class mouseMove:
    def __init__(self, xy_to_uv_mat, zThresh=-0.02):
        self.zThresh = zThresh
        self.tranform = xy_to_uv_mat
        self.lastState = False
        pyautogui.FAILSAFE = False
        for m in get_monitors():
            self.xRes = m.width
            self.yRes = m.height

    def moveMouse(self, position, leftClick=True):
        xy = np.array([position[0], position[1], 1])
        uv = np.matmul(self.tranform, xy)
        x = uv[0]/uv[2]
        y = uv[1]/uv[2]
        z = position[2]
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.xRes:
            x = self.xRes
        if y > self.yRes:
            y = self.yRes
        # if not self.lastState and z > self.zThresh:
        #     pyautogui.mouseDown(x,y,button='left', _pause=False)
        #     self.lastState = True
        # elif self.lastState and z < self.zThresh:
        #     pyautogui.mouseUp(x,y,button='left', _pause=False)
        #     self.lastState = False
        # else:
        pyautogui.moveTo(x, y, _pause=False)
        return
