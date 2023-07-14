import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA

mode = "start"
# mode = "continue"
# mode = "load"
# mode = "configMOO"

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable/output/2022-12-09_07-07-13/ElitePool_55.gacheckpoint.npy"

if mode == "start":
    GASetting = {
        'nGenesPerPool': 64,
        'nSurvivedMin': 16,
        'nGensPerPool': 8,

        'nWorkers': multiprocessing.cpu_count(),

        'folderDir': 'scripts/trainDissolveLobster/',
        'graphMutationChance': 0.1,
        'contractionMutationChance': 0.1,
        'contractionCrossChance': 0.1,
        'actionMutationChance': 0.2,
        'actionCrossChance': 0.2,
        'crossChance': 0.8,
        'randomInit': True
    }
    ga = GA(GASetting=GASetting)
    ga.run()

elif mode == "continue":
    ga = GA(GACheckpointDir=GACheckpointDir)
    ga.run()

elif mode == "load":
    ga = GA(GACheckpointDir=GACheckpointDir)
    print('genePool')
    ga.logPool(ga.genePool, printing=True, showAllGenes=True, showRValue=True)
    print('elitePool')
    ga.logPool(ga.elitePool, printing=True, showAllGenes=True, showRValue=True)

elif mode == "configMOO":
    mooDict = readMooDict('./scripts/trainDissolve/data')

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
