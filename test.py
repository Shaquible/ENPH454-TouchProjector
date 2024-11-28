import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.handPositionFilter import deEmphasis

df = pd.read_csv('dataCapture.csv')
xs = df['X'].to_numpy()*1000
ys = df['Y'].to_numpy()*1000
zs = df['Z'].to_numpy()*1000
deemph = deEmphasis()
for i in range(len(xs)):
    xs[i], ys[i], zs[i] = deemph.smoothPos(np.array([xs[i], ys[i], zs[i]]))
times = df['Time'] - df['Time'][0]
frames = np.arange(len(xs))
dts = np.diff(times)
print(np.mean(dts), np.std(dts))
plt.plot(frames, xs, label='X')
plt.plot(frames, ys, label='Y')
plt.plot(frames, zs, label='Z')
plt.legend()
plt.show()
