import pyautogui
from screeninfo import get_monitors
class mouseMove:
    def __init__(self, threshold=0.02, xlen = 0.30, ylen = 0.21):
        self.threshold = threshold
        self.xlen = xlen
        self.ylen = ylen
        pyautogui.FAILSAFE = False
        
        for m in get_monitors():
            self.xRes = m.width
            self.yRes = m.height
    
    def moveMouse(self, position):
        x = -(position[0])/self.xlen*self.xRes
        y = (position[1])/self.ylen*self.yRes
        x = x + self.xRes/2
        y = y + self.yRes/2
        print(x,y)
        pyautogui.moveTo(x,y)
        if position[2] < self.threshold:
            pyautogui.click()
        return