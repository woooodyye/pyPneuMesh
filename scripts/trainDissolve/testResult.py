import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.MultiMotion import MultiMotion
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA


# mooDict = readNpy('scripts/trainDissolve/output/ElitePool_120.gacheckpoint.npy')
mooDict = readNpy('scripts/trainDissolve/output/0630/ElitePool_108.gacheckpoint.npy')


elitePools = mooDict['elitePoolMOODict']

print('{:20s} {:20s} {:20s} {:20s} {:20s}'.format('total Move upward', 'left', 'right', 'up', 'down'))
for i in range(len(elitePools)):
    elite = elitePools[i]
    moo = elite['mooDict']
    # breakpoint()
    # model = moo.model
    score = elite['score']
    print('{:20f} {:20f} {:20f} {:20f} {:20f}'.format(score[0],score[1],score[2],score[3],score[4]))

#4 move forward
#1 move left 

#8 move right

#0 move backward

# moo = elitePools[4]['mooDict']
# m = Model(moo['trussParam'], moo['simParam'])
# # m.edgeChannel[m.edgeActive==False] = -1
# mm= MultiMotion(moo['actionSeqs'], m)
# vs, _ = mm.simulate(0, 4, False)
# m.animate(vs, 5)


# moo = elitePools[1]['mooDict']
# m = Model(moo['trussParam'], moo['simParam'])
# m.edgeChannel[m.edgeActive==False] = -1
# mm= MultiMotion(moo['actionSeqs'], m)
# vs, _ = mm.simulate(0, 2, True)
# m.animate(vs, 5)