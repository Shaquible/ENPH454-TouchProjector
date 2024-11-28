import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt("handTrackerResults.csv", delimiter=",")
plt.plot(data[:, 0], data[:, 1], "ro")
plt.xlabel("Number of Pixels")
plt.ylabel("Time (s)")
plt.show()
