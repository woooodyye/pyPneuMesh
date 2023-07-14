import numpy as np

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.MultiMotion import MultiMotion
from pyPneuMesh.MultiObjective import MultiObjective
from pyPneuMesh.MOO import MOO
# trussParam = readNpy('scripts/testGrab/data/lobster_grab_new.trussparam.npy')
# simParam = readNpy('scripts/testGrab/data/lobster.simparam.npy')
# actionSeqs = readNpy('scripts/testGrab/data/lobster_grab_new.actionseqs.npy')
# # objectives = readNpy('scripts/testDissolveMultiObjective/lobster/lobster.objectives.npy')
# graphSetting = readNpy('scripts/testGrab/data/lobster.graphsetting.npy')
# # objectives[0]['dissolve']= False
# # objectives[1]['dissolve']= True

# graphSetting['symmetric'] = False
# graphSetting['dissolve'] = True
# graphSetting['iesSubs'] = [
#     [6, 8, 28, 34, 35, 36, 41, 46, 48, 49, 50, 51, 52, 64, 66, 67, 72, 74, 77, 85, 100, 101, 106, 107, 108, 109, 111, 116, 120, 121, 126, 131, 133],
#     [2, 4, 7, 10, 14, 21, 24, 29, 32, 33, 37, 40, 42, 43, 47, 62, 65, 71, 78, 79, 83, 95, 96, 98, 99, 104, 105, 110, 115, 117, 118, 124, 127],
#     [5, 9, 13, 15, 16, 17, 18, 20, 22, 23, 25, 27, 30, 31, 53, 54, 55, 57, 58, 59, 60, 61, 63, 68, 69, 73, 75, 76, 81, 82, 84, 86, 87, 88, 89, 91, 92, 94, 97, 112, 113, 114, 122, 123, 128, 129, 132, 134, 135, 136]
# ]
# graphSetting['numChannels'] = 3
# graphSetting['groups'] = 3
# np.save('scripts/trainDissolveLobster/data/lobster.graphSetting.npy',graphSetting)

# m = Model(trussParam, simParam)
# m.show()
# mm = MultiMotion(actionSeqs, m)


    # action0 = np.zeros((4,8), dtype = int)
    # action1 = np.zeros((2,8), dtype = int)

    # obj0 = {'numLoop' : 4, 'subObjectives' : ['moveUpward'], 'dissolve' : False}
    # obj1 = {'numLoop' : 2, 'subObjectives' : ['leftClawMove'], 'dissolve' : True}
    # obj2 = {'numLoop' : 2, 'subObjectives' : ['rightClawMove'], 'dissolve' : True}
    # obj3 = {'numLoop' : 2, 'subObjectives' : ['headMove'], 'dissolve' : True}
    # obj4 = {'numLoop' : 2, 'subObjectives' : ['bodyMove'], 'dissolve' : True}

    # mooDict['actionSeqs'] = {0 : action0, 1 : action1, 2: action1, 3: action1, 4: action1}
    # mooDict['objectives'] = {0 : obj0, 1 : obj1, 2 : obj2, 3: obj3, 4: obj4}

    # mooDict['graphSetting']  = {'symmetric' : False, 'dissolve' : True}
    # moo = MOO(mooDict=mooDict)

    # moo.saveAll('scripts/trainDissolve/data', 'lobster')

mooDict = readMooDict('./scripts/trainDissolveLobster/data')

# objectives = readNpy('scripts/trainDissolveLobster/data/lobster.objectives.npy')
# action0 = np.zeros((1,9), dtype = int)
# action1 = np.zeros((5,9), dtype = int)

# obj0 = {'numLoop' : 1, 'subObjectives' : ['grabLobster'], 'dissolve' : False}
# obj1 = {'numLoop' : 5, 'subObjectives' : ['moveLeft', 'faceLeft'], 'dissolve' : False}
# obj2 = {'numLoop' : 5, 'subObjectives' : ['leftAway'], 'dissolve' : True}
# obj3 = {'numLoop' : 5, 'subObjectives' : ['rightAway'], 'dissolve' : True}

# mooDict['actionSeqs'] = {0 : action0, 1 : action1, 2: action1, 3: action1}
# mooDict['objectives'] = {0 : obj0, 1 : obj1, 2 : obj2, 3: obj3}

# moo = MOO(mooDict=mooDict)
# moo.saveAll('scripts/trainDissolveLobster/data', 'lobster')

graphSetting = readNpy('scripts/trainDissolveLobster/data/lobster.graphsetting.npy')
graphSetting['symmetric'] = False
graphSetting['dissolve'] = True
graphSetting['iesSubs'] = [
    [6, 8, 28, 34, 35, 36, 41, 46, 48, 49, 50, 51, 52, 64, 66, 67, 72, 74, 77, 85, 100, 101, 106, 107, 108, 109, 111, 116, 120, 121, 126, 131, 133],
    [2, 4, 7, 10, 14, 21, 24, 29, 32, 33, 37, 40, 42, 43, 47, 62, 65, 71, 78, 79, 83, 95, 96, 98, 99, 104, 105, 110, 115, 117, 118, 124, 127],
    [5, 9, 13, 15, 16, 17, 18, 20, 22, 23, 25, 27, 30, 31, 53, 54, 55, 57, 58, 59, 60, 61, 63, 68, 69, 73, 75, 76, 81, 82, 84, 86, 87, 88, 89, 91, 92, 94, 97, 112, 113, 114, 122, 123, 128, 129, 132, 134, 135, 136]
]
graphSetting['numChannels'] = 3
graphSetting['groups'] = 3
np.save('scripts/trainDissolveLobster/data/lobster.graphSetting.npy',graphSetting)
