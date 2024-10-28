#This code tests the handPositionFilter code to see if it is actually smoothing the data well.
import numpy as np
import matplotlib.pyplot as plt
from handPositionFilter import rollingAvg_PID

N = 10000
handPosition = rollingAvg_PID([0.1, 0.1, 0.001],2,2,3)

testData = np.zeros(N)
x = np.linspace(0,N-1,num = N)
testData = (1/N**2)*(x**2)
noise = 0.1*np.random.rand(N,1)
testOutput = testData + noise
for i in range(3,N):
    inputData = testOutput[i]
    imgPos = testData[i-2]
    filteredData = handPosition.smoothPos(inputData, imgPos, 0.01)
    
plt.plot(x,testData, 'rx')
plt.plot(x,filteredData, 'bo')