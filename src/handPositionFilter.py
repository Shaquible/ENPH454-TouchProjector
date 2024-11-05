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

    def smoothPos(self, pos):
        self.xs[self.xind] = pos[0]
        self.ys[self.yind] = pos[1]
        self.zs[self.zind] = pos[2]
        outPos = [np.mean(self.xs), np.mean(self.ys), np.mean(self.zs)]
        self.xind = (self.xind + 1) % self.x_len
        self.yind = (self.yind + 1) % self.y_len
        self.zind = (self.zind + 1) % self.z_len
        return outPos


class lowPassVelocity:
    def __init__(self, xWindow, yWindow, zWindow):
        self.prev = np.zeros(3)
        self.prevOut = np.zeros(3)
        self.prevTime = time.time()
        self.vXs = np.zeros(xWindow)
        self.vYs = np.zeros(yWindow)
        self.vZs = np.zeros(zWindow)
        self.xind = 0
        self.yind = 0
        self.zind = 0
        self.xWindow = xWindow
        self.yWindow = yWindow
        self.zWindow = zWindow

    def smoothPos(self, pos):
        # update velocity based on the change in position measured
        t = time.time()
        dt = t - self.prevTime
        self.prevTime = t
        vel = (pos - self.prev)/dt
        self.prev = pos
        self.vXs[self.xind] = vel[0]
        self.vYs[self.yind] = vel[1]
        self.vZs[self.zind] = vel[2]
        outVel = np.array(
            [np.mean(self.vXs), np.mean(self.vYs), np.mean(self.vZs)])
        self.xind = (self.xind + 1) % self.xWindow
        self.yind = (self.yind + 1) % self.yWindow
        self.zind = (self.zind + 1) % self.zWindow
        # update position based on the velocity and the previous position
        self.prevOut = self.prevOut + outVel*dt
        return self.prevOut


class deEmphasis:
    def __init__(self, coef=0.95):
        self.coef = coef
        self.prev = np.zeros(3)

    def smoothPos(self, pos):
        out = (pos + self.prev*(self.coef))/(1+self.coef)
        self.prev = out
        return out


# class SavitzkyGolay:
#     def __init__(self, window):
#         self.filtered = 0
#         self.windowLength = window
#         self.stored = np.zeros((self.n_avg,3))
#         self.index = 0

#     def smoothPos(self,pos):
#         self.stored[self.index,:] = pos
#         self.filtered = savgol_filter(self.stored, self.windowLength, 2)
#         if self.index < self.n_avg - 1:
#             self.index = self.index + 1
#         else:
#             self.index = 0
