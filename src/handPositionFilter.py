#Class to control the moving average
import numpy as np
from scipy.filter import savgol_filter

class rollingAvg:
    def __init__(self, n_avg):
        # Defines the rolling average of the hand tracked position.
        self.average = 0
        self.n_avg = n_avg
        self.stored = np.zeros((n_avg,3))
        self.index = 0
        
    def smoothPos(self,pos):
        self.stored[self.index,:] = pos
        self.average = np.sum(self.stored, axis=0)/self.n_avg
        if self.index < self.n_avg - 1:
            self.index = self.index + 1
        else:
            self.index = 0
            
        return self.average
            
            
class SavitzkyGolay:
    def __init__(self, window):
        self.filtered = 0
        self.windowLength = window
        self.stored = np.zeros((n_avg,3))
        self.index = 0
        
    def smoothPos(self,pos):
        self.stored[self.index,:] = pos
        self.filtered = savgol_filter(self.stored, self.windowLength, 2)
        if self.index < self.n_avg - 1:
            self.index = self.index + 1
        else:
            self.index = 0
    