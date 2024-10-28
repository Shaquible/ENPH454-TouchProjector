#Class to control the moving average
import numpy as np
from scipy.signal import savgol_filter

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
        self.pos = [0,0,0]
        
    def smoothPos(self,pos):
        self.xs[self.xind] = pos[0]
        self.ys[self.yind] = pos[1]
        self.zs[self.zind] = pos[2]
        self.pos = [np.mean(self.xs), np.mean(self.ys), np.mean(self.zs)]
        self.xind = (self.xind + 1)%self.x_len
        self.yind = (self.yind + 1)%self.y_len
        self.zind = (self.zind + 1)%self.z_len    
        return self.pos
        
class rollingAvg_PID:
    def __init__(self, tunMtx, x_len, y_len, z_len):
        # Defines the rolling average of the hand tracked position.
        self.average = 0
        self.x_len = x_len
        self.xs = np.zeros(x_len)
        self.xerrs = np.zeros(x_len)
        self.y_len = y_len
        self.ys = np.zeros(y_len)
        self.yerrs = np.zeros(y_len)
        self.z_len = z_len
        self.zs = np.zeros(z_len)
        self.zerrs = np.zeros(z_len)
        self.xind = 0
        self.yind = 0
        self.zind = 0
        self.tunMtx = tunMtx
        self.pos = [0,0,0]
        
    def der(vals, ind):
        return 0.5*(vals[ind] - vals[ind-2])
        
    # This smooth position function includes a simple PID control algorithm as well to further smooth the system and rapid changes to the system.
    def smoothPos(self,pos,imgPos, timeStep):
        self.xs[self.xind] = pos[0]
        self.ys[self.yind] = pos[1]
        self.zs[self.zind] = pos[2]
        filteredPos = [np.mean(self.xs), np.mean(self.ys), np.mean(self.zs)]
        
        self.xerrs[self.xind] =  filteredPos[0] - imgPos[0]
        self.yerrs[self.yind] =  filteredPos[1] - imgPos[1]
        self.zerrs[self.zind] =  filteredPos[2] - imgPos[2]
        
        xI = self.xI + self.xerrs[self.xind]*timeStep
        yI = self.yI + self.yerrs[self.yind]*timeStep
        zI = self.zI + self.zerrs[self.zind]*timeStep
        
        xD = (self.xerrs[self.xind] - self.xerrs[(self.xind-1)%self.x_len])/timeStep
        yD = (self.yerrs[self.yind] - self.yerrs[(self.yind-1)%self.y_len])/timeStep
        zD = (self.zerrs[self.zind] - self.zerrs[(self.zind-1)%self.z_len])/timeStep
        
        outPos = [0,0,0]
        outPos[0] = self.tunMtx[0]*self.xerrs[self.xind] + self.tunMtx[1]*xI + self.tunMTX[2]*xD
        outPos[1] = self.tunMtx[0]*self.yerrs[self.yind] + self.tunMtx[1]*yI + self.tunMTX[2]*yD
        outPos[2] = self.tunMtx[0]*self.zerrs[self.zind] + self.tunMtx[1]*zI + self.tunMTX[2]*zD
        
        self.xind = (self.xind + 1)%self.x_len
        self.yind = (self.yind + 1)%self.y_len
        self.zind = (self.zind + 1)%self.z_len
        
        return outPos
            
# class SavitzkyGolay:
#     def __init__(self, x_len, y_len, z_len):
#         self.x_len = x_len
#         self.xs = np.zeros(x_len)
#         self.y_len = y_len
#         self.ys = np.zeros(y_len)
#         self.z_len = z_len
#         self.zs = np.zeros(z_len)
#         self.xind = 0
#         self.yind = 0
#         self.zind = 0
        
#     def smoothPos(self,pos):
#         self.xs[self.xind] = pos[0]
#         self.ys[self.yind] = pos[1]
#         self.zs[self.zind] = pos[2]
        
#         self.xind = (self.xind + 1)%self.x_len
#         self.yind = (self.yind + 1)%self.y_len
#         self.zind = (self.zind + 1)%self.z_len    
#         return outPos
    

    