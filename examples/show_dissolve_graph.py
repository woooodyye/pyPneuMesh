from utils.moo import MOO
# from utils.objectives.locomotion import MoveForward, FaceForward, TurnLeft, LowerBodyMax
from utils.objectives.transform import FrontAlign, SideAlign
MOOsetting = {
    'modelDir': './data/lobster.json',
    'numChannels': 3,
    'numActions': 2,
    'numObjectives': 2,
    'numTargets': 1,
    "channelMirrorMap": {
        0: 1,
        2: -1,
    },
    'objectives': [[SideAlign],[FrontAlign]],
    'meshDirs': ['./data/half_helmet_new_mesh.json','./data/half_helmet_new_mesh.json'],
    'keyPointsIndices': [9, 22, 23, 52,53,54, 55]
}

moo = MOO(MOOsetting, randInit= True)
model = moo.model


model.initSplitGraph()
model.fromSplitGraph()

moo.simulate()
# model.show()

# # model.initGraph()
# # model.show()

# for i in range(5):
#     # model.mutateHalfGraph()
#     # model.mutateGraph()
#     model.mutate()
#     model.show()
