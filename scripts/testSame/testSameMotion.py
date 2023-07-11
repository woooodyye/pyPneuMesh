from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.SplitGraph import SplitGraph
from pyPneuMesh.MultiMotion import MultiMotion

import numpy as np
leftClaw = readNpy('scripts/testSame/lobster_left/lobster_left.trussparam.npy')
simParam = readNpy('scripts/testSame/lobster/lobster.simparam.npy')
leftActions = readNpy('scripts/testSame/lobster_left/lobster_left.actionseqs.npy')



L = np.array([59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 84, 85, 86, 87, 88, 89, 90, 91, 92],dtype=int)
eSubs =[29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 45, 46, 47, 48, 49, 
                      50, 51, 52, 53, 129, 130, 131,
                     59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                     95, 96, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                       110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                         122, 123, 124, 125, 126, 127, 128, 133, 134]#

A = np.arange(0, 135)

eNot =np.array([ 21,  22,  23,  24,  25,  26,  27,  28,  30,  54,  55,  56,  57,
        58,  61,  93,  94,  97, 100, 132],dtype= int)

left_m = Model(leftClaw, simParam)

mm = MultiMotion(leftActions, left_m)



full = readNpy('scripts/testSame/lobster/lobster_full.trussparam.npy')
full_m = Model(full, simParam)

lobster_dis = readNpy('scripts/testSame/lobster/lobster_dis.trussparam.npy')
dis_m = Model(lobster_dis, simParam)

def change_indices(array, value, replacement):
    mask = np.isin(array, value)
    print(mask)
    array[mask] = replacement
    return array

# Example usage
result = change_indices(A, eSubs, 0)

full_m.edgeChannel[eNot] = 0 #test to see
full_m.edgeActive[eNot] = False
full_m.maxLengths[full_m.edgeActive] = full_m.MAX_ACTIVE_BEAM_LENGTH


mm_full = MultiMotion(leftActions, full_m)
mm_dis = MultiMotion(leftActions, dis_m)


vs_left, _ = mm.simulate(0, 10)
vs, _ = mm_full.simulate(0, 10, True) #dissolve 


vs_non, _ = mm_dis.simulate(0, 10, False)

ivs = sorted(set(full_m.e[L].reshape(-1).tolist()))



#NEW TEST
dissolve = SplitGraph(dis_m)
dissolve.randomize()
dissolve.toModel()
m = dissolve.model
A = np.zeros((2,8),dtype= int)
A[0][0]= 1
leftActions = {0 : A}
mm_dis = MultiMotion(leftActions, m)
vs, _ = mm_dis.simulate(0, 10, True)
vs_non,_ = mm_dis.simulate(0,10, False)