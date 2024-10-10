#This code tests the handPositionFilter code to see if it is actually smoothing the data well.
import numpy as np
from handPositionFilter import rollingAvg

N = 10000
handPosition = rollingAvg(30)

testData = np.random.rand(N,3)
for i in range(N):
    inputData = testData[i,:]
    handPosition.addPos(inputData)
    print(handPosition.average)
    