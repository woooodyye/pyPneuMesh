from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.SplitGraph import SplitGraph
from pyPneuMesh.MultiMotion import MultiMotion
import numpy as np

trussParam = readNpy('scripts/testDissolve/lobster/lobster.trussparam.npy')
simParam = readNpy('scripts/testDissolve/lobster/lobster.simparam.npy')
actionSeqs = readNpy('scripts/testDissolve/lobster/lobster.actionseqs.npy')
halfGraphSetting = readNpy('scripts/testDissolve/lobster/lobster.graphsetting.npy')


m = Model(trussParam, simParam)
dissolve = SplitGraph(m)
dissolve.randomize()
dissolve.toModel()

# print(actionSeqs)
actionSeqs[0]= np.ones((2,8))
actionSeqs[1] = np.zeros((8,2))
mm = MultiMotion(actionSeqs, m)


vs = mm.simulate(0, 2, False)
vs = mm.animate(0, 2)

