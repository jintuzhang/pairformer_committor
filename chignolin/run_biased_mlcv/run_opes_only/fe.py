import plumed
import numpy as np

for i in range(1, 2000):
    f1 = plumed.read_as_pandas('./grids/ff_{}.dat'.format(i))
    f11 = f1[f1['z_fitted'] < 0.8]
    f12 = f1[f1['z_fitted'] > 0.8]
    fesA = -2.83 * np.logaddexp.reduce(-1 / 2.83 * f12['file.free'])
    fesB = -2.83 * np.logaddexp.reduce(-1 / 2.83 * f11['file.free'])
    print(i * 10, fesB - fesA)
