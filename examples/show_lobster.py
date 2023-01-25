import pickle5
import numpy as np
from utils.mesh import Mesh
from utils.truss import Truss
from utils.geometry import boundingBox, bboxDiagonal, center, translationMatrix, scaleMatrix, transform3d

# result = pickle5.load(open('./output/lobster_syn/iPool_12', 'rb'))
result = pickle5.load(open('./output/lobster_1205_biasfar/iPool_64', 'rb'))

# print('{:20s} {:20s} {:20s} {:20s}'.format('move forward', 'move forward', 'grab lobster', 'minEnergy'))
# for i in range(len(result['elitePool'])):
#     elite = result['elitePool'][i]
#     moo = elite['moo']
#     model = moo.model
#     score = elite['score']
#     print('{:20f} {:20f} {:20f} '.format(score[0], score[1], score[2]))

# moo_energy = result['elitePool'][-3]['moo']  # minimize energy

moo_action = result['elitePool'][1]['moo']
moo_energy = result['elitePool'][2]['moo']

actionSeq = moo_action.actionSeqs[0]

moo_action.simulate(actionSeq, nLoops=2, visualize=True)
# moo_energy.simulate(actionSeq_energy, nLoops=2, visualize=True)

# actionSeq = moo_action.actionSeqs[0]  # control sequence of the second objective
# actionSeq_energy = moo_action.actionSeqs[1]  # control sequence of the second objective
#
# # moo_action.simulate(actionSeq, nLoops=2, visualize=False)  # visualize the trajectory of the control
# # print(moo_action.model.vEnergy.sum() / score[0])
# moo_action.simulate(actionSeq_energy, nLoops=2, visualize=False)  # visualize the trajectory of the control
# print(moo_action.model.vEnergy.sum() / score[1])

# moo_energy.simulate(actionSeq_energy, nLoops=2, visualize=True)  # visualize the trajectory of the control

# moo_action.simulate(moo_action.actionSeqs[0], nLoops=2, visualize=False)

# moo_action.simulate(moo_action.actionSeqs[1], nLoops=2, visualize=False)
