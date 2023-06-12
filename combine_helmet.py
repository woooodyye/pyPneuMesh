from utils.moo import MOO
# from utils.objectives.objective import objMoveForward, objFaceForward, objTurnLeft, objTurnRight, objLowerBodyMax
from utils.objectives.transform import KeyPointsAlign
import numpy as np
import pickle5

result_A =  pickle5.load(open('./output/iPool_126', 'rb'))
result_B = pickle5.load(open('./output/iPool_242', 'rb'))

index_A = 0 #which gene from elitePool A
index_B = 0 #which gene from elitePool B

elite_A = result_A['elitePool'][index_A]
elite_B = result_B['elitePool'][index_B]

moo_A = elite_A['moo']
moo_B = elite_B['moo']
model_A=  moo_A.model
model_B = moo_B.model

#in place update
moo_B.model.edgeChannel[model_B.edgeActive] +=2

moo_A.model.edgeActive =  np.logical_or(model_A.edgeActive, model_B.edgeActive)
moo_A.model.edgeChannel = moo_A.model.edgeChannel  + moo_B.model.edgeChannel
actionSeqA = moo_A.actionSeqs[0]
actionSeqB = moo_B.actionSeqs[0]
# Get the shape of the input arrays
rows1, cols1 = actionSeqA.shape
rows2, cols2 = actionSeqB.shape

# Calculate the new shape for the output array
new_rows = rows1 + rows2
new_cols = cols1 + cols2

# Create a zero-filled array with the new shape
result = np.zeros((new_rows, new_cols), dtype=actionSeqA.dtype)

# Fill the diagonal with array1
result[:rows1, :cols1] = actionSeqA

# Fill the diagonal with array2
result[rows2:, cols2:] = actionSeqB


moo_A.simulate(result, nLoops = 4, visualize = True)
# moo = result['elitePool'][0]['moo']
# # moo.model.show()  # visualize the truss static shape and channels

# moo_A.model.exportJSON()

# actionSeq = moo.actionSeqs[0]  # control sequence of the second objective
# moo.simulate(actionSeq, nLoops=4, visualize=True)  # visualize the trajectory of the control
