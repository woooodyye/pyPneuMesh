# from utils.objectives.objective import objMoveForward, objFaceForward, objTurnLeft, objLowerBodyMax
from utils.GA import GeneticAlgorithm
import argparse
import numpy as np
import multiprocessing

from utils.moo import MOO
from utils.objectives.transform import KeyPointsAlign, SurfaceAlign

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='The directory of the checkpoint file.')
args = parser.parse_args()

MOOsetting = {
    'modelDir': './data/helmet_inflated.json',
    'numChannels': 4,
    'numActions': 1,
    'numObjectives': 1,
    'numTargets': 1,
    'objectives': [[KeyPointsAlign]],
    'meshDirs': ['./data/half_helmet_mesh.json'],
    'keyPointsIndices': [9, 22, 23],
    'nLoopSimulate': 1
}

moo = MOO(MOOsetting, randInit=False)

print(moo.model.v[9])
print(moo.model.v[22])
print(moo.model.v[23])

actionSeq0 = moo.actionSeqs[0]  # control sequence of the second objective
print(actionSeq0)
actionSeq0 = np.array([[False], [False], [False], [False], [False]])

model = moo.model

# test
# model.toHalfGraph(reset=True)
# model.fromHalfGraph()
# model.initHalfGraph()

# model.show()

vs, es = moo.simulate(actionSeq0, nLoops=20, visualize=False, mesh=None)
print(vs.shape)
print(vs[0])
vss = np.tile(vs, (10, 1, 1))
print(vss.shape)
