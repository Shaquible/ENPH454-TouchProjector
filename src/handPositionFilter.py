#Class to control the moving average
import numpy as np

class rollingAvg:
    def __init__(self, n_avg):
        # Defines the rolling average of the hand tracked position.
        self.average = 0
        self.n_avg = n_avg
        self.stored = np.zeros((n_avg,3))
        self.index = 0
        
    def addPos(self,pos):
        self.stored[self.index,:] = pos
        self.average = np.sum(self.stored, axis=0)
        if self.index < self.n_avg - 1:
            self.index = self.index + 1
        else:
            self.index = 0
                

        
        
    