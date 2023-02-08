import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA

dict = readMooDict('scripts/trainTable_64/data')

# mode = "start"
# mode = "continue"
# mode = "load"
mode = "configMOO"

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable/output/2022-12-09_07-07-13/ElitePool_55.gacheckpoint.npy"

if mode == "start":
    GASetting = {
        'nGenesPerPool': 64,
        'nSurvivedMin': 16,
        'nGensPerPool': 8,

        'nWorkers': multiprocessing.cpu_count(),

        'folderDir': 'scripts/trainTable_64/',

        'graphMutationChance': 0.1,
        'contractionMutationChance': 0.1,
        'contractionCrossChance': 0,
        'actionMutationChance': 0.2,
        'actionCrossChance': 0,
        'crossChance': 0,
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
    mooDict = readMooDict('./scripts/trainTable_64/data')
    moo = MOO(mooDict=mooDict)

    m = Model(mooDict['trussParam'], mooDict['simParam'])

    d = {}
    for i in range(0, 63, 2):
        d[i] = i + 1;
