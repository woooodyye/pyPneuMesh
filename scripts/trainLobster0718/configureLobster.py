import numpy as np

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.MultiMotion import MultiMotion
from pyPneuMesh.MultiObjective import MultiObjective
from pyPneuMesh.MOO import MOO

mooDict = readMooDict('./scripts/trainLobster0718/trainGrabLobsterMoveTailAway/data')

# action0 = np.zeros((4,9), dtype = int)
# # action1 = np.zeros((5,9), dtype = int)

# obj0 = {'numLoop' : 5, 'subObjectives' : ['holdClaw','moveLeft'], 'dissolve' : False}
# # obj1 = {'numLoop' : 5, 'subObjectives' : ['moveLeft', 'faceLeft'], 'dissolve' : False}
# obj1 = {'numLoop' : 5, 'subObjectives' : ['tailAway'], 'dissolve' : True}
# # obj3 = {'numLoop' : 5, 'subObjectives' : ['rightAway'], 'dissolve' : True}

# # mooDict['actionSeqs'] = {0 : action0, 1 : action1, 2: action1, 3: action1}
# # mooDict['objectives'] = {0 : obj0, 1 : obj1, 2 : obj2, 3: obj3}

# mooDict['actionSeqs'] = {0 : action0, 1: action0}
# mooDict['objectives'] = {0 : obj0, 1: obj1}
# moo = MOO(mooDict=mooDict)
# moo.saveAll('scripts/trainLobster0718/trainGrabLobsterMoveTailAway/data', 'lobster')
