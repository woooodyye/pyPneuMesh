# from utils.objectives.objective import objMoveForward, objFaceForward, objTurnLeft, objLowerBodyMax
from utils.GA import GeneticAlgorithm
import argparse
import multiprocessing

from utils.objectives.transform import KeyPointsAlign, SurfaceAlign

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='The directory of the checkpoint file.')
args = parser.parse_args()

MOOsetting = {
    'modelDir': './data/half_helmet.json',
    'numChannels': 4,
    'numActions': 1,
    'numObjectives': 1,
    'numTargets': 1,
    "channelMirrorMap": {
        0: 1,
        2: -1,
        3: -1
    },
    'objectives': [[KeyPointsAlign]],
    'meshDirs': ['./data/half_helmet_mesh.json'],
    'keyPointsIndices': [9, 22, 23],
    'nLoopSimulate': 1
}

nWorkers = multiprocessing.cpu_count()
nGensPerPool = int(nWorkers / 8 * 5)
settingGA = {
    'nGenesPerPool': nWorkers,
    'nGensPerPool': int(nWorkers / 8 * 5),
    'nSurvivedMax': int(nGensPerPool * 0.5),

    'nWorkers': nWorkers,
    'plot': True,
    'mute': False,
    'saveHistory': True,
}

ga = GeneticAlgorithm(MOOSetting=MOOsetting, GASetting=settingGA)

if args.checkpoint:
    checkpointDir = args.checkpoint
    ga.loadCheckpoint(checkpointDir)

ga.run()
