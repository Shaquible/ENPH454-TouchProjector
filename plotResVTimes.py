import matplotlib.pyplot as plt
import numpy as np
data1 = np.loadtxt("handTrackerResults.csv", delimiter=",")
data2 = np.loadtxt("handTrackerResultsWithHSVConversion.csv", delimiter=",")
# find the slope of the line
m1 = np.mean(np.diff(data1[:, 1])/np.diff(data1[:, 0]))
m2 = np.mean(np.diff(data2[:, 1])/np.diff(data2[:, 0]))
print(m1, m2)
plt.plot(data1[:, 0], data1[:, 1]*1000, "ro", label="Inference")
plt.plot(data2[:, 0], data2[:, 1]*1000, "bo",
         label="Inference + Image Recoloring")
plt.text(0, 0.5, "Inference Slope: " + str(m1), fontsize=12)
plt.text(0, 0.4, "Inference + Image Recoloring Slope: " + str(m2), fontsize=12)
plt.xlabel("Number of Pixels")
plt.ylabel("Frame Time (ms)")
plt.legend()
plt.savefig("handTrackerResults.png")
