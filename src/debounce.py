import numpy as np

class Debouncer:
    def __init__(self, winLength, initialState) -> None:
        self.N = winLength
        self.idx = 0
        self.buffer = np.zeros(self.N) + initialState
        self.prevState = initialState
        return
    def debounce(self, val):
        self.buffer[self.idx] = val
        self.idx = (self.idx + 1) % self.N
        if np.all(self.buffer == self.buffer[0]):
            self.buffer[0]
            self.prevState = self.buffer[0]
        return self.prevState
        