from utils.GA import GeneticAlgorithm
from utils.objectives.locomotion import MoveForward, FaceForward, TurnLeft, LowerBodyMax
from utils.objectives.energy import MinEnergy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='The directory of the checkpoint file.')
args = parser.parse_args()
MOOsetting = {
    'modelDir': './data/lobster.json',
    'numChannels': 4,
    'numActions': 4,
    'numObjectives': 2,
    "channelMirrorMap": {
        0: 1,
        2: -1,
        3: -1
    },
    'objectives': [[MoveForward], [MoveForward, MinEnergy]],
    'nLoopSimulate': 2
}
nWorkers = 1
nGensPerPool = 1
settingGA = {
    'nGenesPerPool': nWorkers,
    'nGensPerPool': nGensPerPool,
    'nSurvivedMax': nGensPerPool,

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
