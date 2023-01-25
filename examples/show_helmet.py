import pickle5

from utils.mesh import Mesh
from utils.truss import Truss
from utils.geometry import boundingBox, bboxDiagonal, center, translationMatrix, scaleMatrix, transform3d

result = pickle5.load(open('./output/1115_helmet/iPool_69', 'rb'))

print('{:20s} {:20s} {:20s} {:20s}'.format('keypointsAlign', 'face forward', 'turn left', 'lower height'))
for i in range(len(result['elitePool'])):
    elite = result['elitePool'][i]
    moo = elite['moo']
    model = moo.model
    score = elite['score']
    print('{:20f}'.format(score[0]))

moo = result['elitePool'][0]['moo']
# moo.model.show()  # visualize the truss static shape and channels

actionSeq0 = moo.actionSeqs[0]  # control sequence of the second objective
print(actionSeq0)
# actionSeq1 = moo.actionSeqs[1]
# assert (actionSeq0.all() == actionSeq1.all())
moo.nLoopSimulate = 10

# moo.model.show()

# test it out ...
mesh = Mesh('./data/half_helmet_mesh.json', boundingBox(moo.model.v))

moo.simulate(actionSeq0, nLoops=1, visualize=False, mesh=None)  # visualize the trajectory of the control
