import numpy as np
import matplotlib.pyplot as plt

from pymoo.indicators.hv import HV

from pyPneuMesh.utils import readNpy

a = readNpy('scripts/trainTable_16/output/2023-02-23_19-02-40/ElitePool_1300.gacheckpoint.npy')

ref_point = np.array([20.0, 0])

ind = HV(ref_point=ref_point)
x = []
y = []

A = np.arange(0, 20, 0.1)
B = np.arange(-20, 0, 0.1)
for i in range(200):
    scores = np.array([A[i],B[i]])
    x.append(i)
    y.append(ind(scores) )


plt.plot(x, y, '-o')

A = np.arange(0, 20, 0.01)
B = np.arange(-20, 0, 0.01)
for i in range(200):
    scores = np.array([A[i],B[i]])
    x.append(i)
    y.append(ind(scores))


plt.plot(x, y, '-o')

plt.savefig("dummy.png")


