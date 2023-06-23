import numpy as np

from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.MultiMotion import MultiMotion
from pyPneuMesh.MultiObjective import MultiObjective

trussParam = readNpy('scripts/trainDissolve/data/lobster.trussparam.npy')
simParam = readNpy('scripts/trainDissolve/data/lobster.simparam.npy')
actionSeqs = readNpy('scripts/trainDissolve/data/lobster.actionseqs.npy')
objectives = readNpy('scripts/trainDissolve/data/lobster.objectives.npy')

m = Model(trussParam, simParam)
mm = MultiMotion(actionSeqs, m)
mo = MultiObjective(objectives, mm)


scores = mo.evaluate()

