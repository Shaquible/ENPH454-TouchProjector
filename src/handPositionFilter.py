# Class to control the moving average
import numpy as np
import time
# from scipy.filter import savgol_filter


class rollingAvg:
    def __init__(self, x_len, y_len, z_len):
        # Defines the rolling average of the hand tracked position.
        self.average = 0
        self.x_len = x_len
        self.xs = np.zeros(x_len)
        self.y_len = y_len
        self.ys = np.zeros(y_len)
        self.z_len = z_len
        self.zs = np.zeros(z_len)
        self.xind = 0
        self.yind = 0
        self.zind = 0
        self.pos = [0, 0, 0]

    def smoothPos(self, pos):
        self.xs[self.xind] = pos[0]
        self.ys[self.yind] = pos[1]
        self.zs[self.zind] = pos[2]
        self.pos = [np.mean(self.xs), np.mean(self.ys), np.mean(self.zs)]
        self.xind = (self.xind + 1) % self.x_len
        self.yind = (self.yind + 1) % self.y_len
        self.zind = (self.zind + 1) % self.z_len
        return self.pos


class deEmphasis:
    def __init__(self, coef=0.95):
        self.coef = coef
        self.prev = np.zeros(3)

    def smoothPos(self, pos):
        out = (pos + self.prev*(self.coef))/(1+self.coef)
        self.prev = out
        return out
