import numpy as np

from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.MultiMotion import MultiMotion
from pyPneuMesh.MultiObjective import MultiObjective

trussParam = readNpy('scripts/testDissolveMultiObjective/lobster/lobster_dis.trussparam.npy')
simParam = readNpy('scripts/testDissolveMultiObjective/lobster/lobster.simparam.npy')
actionSeqs = readNpy('scripts/testDissolveMultiObjective/lobster/lobster.actionseqs.npy')
objectives = readNpy('scripts/testDissolveMultiObjective/lobster/lobster.objectives.npy')

objectives[0]['dissolve']= False
objectives[1]['dissolve']= True

m = Model(trussParam, simParam)
mm = MultiMotion(actionSeqs, m)
mo = MultiObjective(objectives, mm)

# # mm.actionSeqs[0] *= 0
# mm.actionSeqs[1] *= 0
# mm.actionSeqs[2] *= 0
# mm.actionSeqs[3] *= 0

scores = mo.evaluate()

