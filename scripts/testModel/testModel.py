import numpy as np

from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model

trussParam = readNpy('scripts/testModel/table/table.trussparam.npy')
simParam = readNpy('scripts/testModel/table/table.simparam.npy')

model = Model(trussParam, simParam)

# breakpoint()
actionSeq = np.ones([3, 3])
actionSeq[1] *= 0
actionSeq[2, 1] *= 0

times, lengths = model.actionSeq2timeNLength(actionSeq)

vs, vEnergys = model.step(50000, times, lengths)

model.show()
model.animate(vs, speed=1.0, singleColor=True)
