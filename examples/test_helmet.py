from utils.moo import MOO
# from utils.objectives.objective import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax
from utils.objectives.transform import KeyPointsAlign

import pickle5
#[52,53,54,55]
result = pickle5.load(open('./output/iPool_126', 'rb'))

# print('{:20s} {:20s} {:20s} {:20s}'.format('move forward', 'face forward', 'turn left', 'lower height'))
for i in range(len(result['elitePool'])):
    elite = result['elitePool'][i]
    moo = elite['moo']
    model = moo.model
    score = elite['score']
    print('{:20f}'.format(score[0]))

moo = result['elitePool'][0]['moo']
# moo.model.show()  # visualize the truss static shape and channels

actionSeq = moo.actionSeqs[0]  # control sequence of the second objective
moo.simulate(actionSeq, nLoops=4, visualize=True)  # visualize the trajectory of the control
