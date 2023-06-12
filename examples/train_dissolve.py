# from utils.objectives.objective import objMoveForward, objFaceForward, objTurnLeft, objLowerBodyMax
from utils.GA import GeneticAlgorithm
import argparse
import multiprocessing

from utils.objectives.transform import KeyPointsAlign, SideAlign, FrontAlign
from utils.objectives.locomotion import MoveUp, MoveLeft, MoveUpward, MoveDown, MoveRight

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='The directory of the checkpoint file.')
args = parser.parse_args()

MOOsetting = {
    'modelDir': './data/lobster.json',
    'numChannels': 9,
    'numActions': 5,
    'numObjectives': 5,
    'numTargets': 1,
    "channelMirrorMap": {
        0: 1,
        2: 3,
        4: 5,
        6: -1,
        7: -1,
        8: -1
    },
    'objectives': [[MoveUpward],[MoveLeft],[MoveRight],[MoveUp],[MoveDown]],
    'meshDirs': ['./data/half_helmet_new_mesh.json','./data/half_helmet_new_mesh.json'],
    'keyPointsIndices': [9, 22, 23, 52,53,54, 55]
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
