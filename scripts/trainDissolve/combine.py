
import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.MultiMotion import MultiMotion
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA


right =[29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 75, 76, 77, 78, 79, 80, 81, 82, 83]
head = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 45, 46, 47, 48, 49, 50, 51, 52, 53, 129, 130, 131]
left = [59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 84, 85, 86, 87, 88, 89, 90, 91, 92]
tail = [95, 96, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 133, 134]

# mooDict = readNpy('scripts/trainDissolve/output/ElitePool_120.gacheckpoint.npy')
mooDict = readNpy('scripts/trainDissolve/output/ElitePool_97.gacheckpoint.npy')


elitePools = mooDict['elitePoolMOODict']

print('{:20s} {:20s} {:20s} {:20s} {:20s} {:20s}'.format('total Move upward',  'faceforward','left', 'right', 'up', 'down'))
for i in range(len(elitePools)):
    elite = elitePools[i]
    moo = elite['mooDict']
    # breakpoint()
    # model = moo.model
    score = elite['score']
    print('{:20f} {:20f} {:20f} {:20f} {:20f} {:20f}'.format(score[0],score[1],score[2],score[3],score[4],score[5]))


moo = elitePools[10]['mooDict']
m_left = Model(moo['trussParam'], moo['simParam'])
action_left= moo['actionSeqs']


moo = elitePools[9]['mooDict']
m_right = Model(moo['trussParam'], moo['simParam'])
action_right = moo['actionSeqs']

moo = elitePools[3]['mooDict']
m_head = Model(moo['trussParam'], moo['simParam'])
action_head = moo['actionSeqs']

moo = elitePools[6]['mooDict']
m_tail = Model(moo['trussParam'], moo['simParam'])
action_tail = moo['actionSeqs']

actions = [[1, 0, 1, 1, 0, 0, 0,1],
           [0, 1, 0, 1, 1, 0, 0,0],
           [1, 0, 0, 1, 0, 1, 0,0],
           [0, 1, 0, 1, 1, 0, 1,0]]

edgeChannel = np.ones_like(m_head.edgeChannel) * -1
contractions = np.ones_like(m_head.contractionLevel) * 7

L = [left, right, head, tail]
M =  [m_left, m_right, m_head, m_tail]
for i in range(len(L)):
    part  = L[i]
    edgeChannel[part] = M[i].edgeChannel[part]
    contractions[part] = M[i].contractionLevel[part]
    

m_tail.edgeChannel = edgeChannel
m_tail.contractionLevel = contractions
moo['actionSeqs'][1] = actions
mm= MultiMotion(moo['actionSeqs'], m_tail)
vs, _ = mm.simulate(1, 4, True)
m_tail.animate(vs, 5)