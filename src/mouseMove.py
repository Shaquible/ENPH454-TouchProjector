import pyautogui
from screeninfo import get_monitors


class mouseMove:
    def __init__(self, zThresh=0.02, xlen=0.30, ylen=0.21):
        self.zThresh = zThresh
        self.xlen = xlen
        self.ylen = ylen
        self.lastState = False
        pyautogui.FAILSAFE = False

        for m in get_monitors():
            self.xRes = m.width
            self.yRes = m.height

    def moveMouse(self, position, leftClick=True):
        x = (position[0])/self.xlen*self.xRes
        y = (position[1])/self.ylen*self.yRes
        z = position[2]
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.xRes:
            x = self.xRes
        if y > self.yRes:
            y = self.yRes
        if not self.lastState and z < self.zThresh:
            pyautogui.mouseDown(button='left', _pause=False)
            self.lastState = True
        elif self.lastState and z > self.zThresh:
            pyautogui.mouseUp(button='left', _pause=False)
            self.lastState = False
        pyautogui.moveTo(x, y, _pause=False)
        return
