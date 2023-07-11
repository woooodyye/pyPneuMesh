import numpy as np

from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.SplitGraph import SplitGraph
from pyPneuMesh.MultiMotion import MultiMotion
from pyPneuMesh.MultiObjective import MultiObjective

trussParam = readNpy('scripts/testDissolve/lobster/lobster.trussparam.npy')
simParam = readNpy('scripts/testDissolve/lobster/lobster.simparam.npy')
actionSeqs = readNpy('scripts/testDissolve/lobster/lobster.actionseqs.npy')
halfGraphSetting = readNpy('scripts/testDissolve/lobster/lobster.graphsetting.npy')
# objectives = readNpy('examples/testMultiObjective/table/table.objectives.npy')

m = Model(trussParam, simParam)
# mo = MultiObjective(objectives, mm)

dissolve = SplitGraph(m)
dissolve.randomize()
dissolve.toModel()

dissolve.model.show()
actionSeqs[0]= np.ones((2,8))
actionSeqs[1] = np.zeros((8,2))
mm = MultiMotion(actionSeqs, dissolve.model)
vs, vEnergys = mm.simulate(0, 2, dissolve=True)

dissolve.model.animate(vs, )