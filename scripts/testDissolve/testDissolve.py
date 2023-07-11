from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.FullGraph import FullGraph
from pyPneuMesh.HalfGraph import HalfGraph
from pyPneuMesh.SplitGraph import SplitGraph
from pyPneuMesh.MultiMotion import MultiMotion

trussParam = readNpy('scripts/testDissolve/lobster/lobster.trussparam.npy')
simParam = readNpy('scripts/testDissolve/lobster/lobster.simparam.npy')
actionSeqs = readNpy('scripts/testDissolve/lobster/lobster.actionseqs.npy')
halfGraphSetting = readNpy('scripts/testDissolve/lobster/lobster.graphsetting.npy')


m = Model(trussParam, simParam)


# m.show()
print(m.contractionLevel)

# halfGraphSetting['mirrorPlane'] = 'y'
# print(halfGraphSetting)
# halfGraph = HalfGraph(m, halfGraphSetting)
# # halfGraph.randomize()
# m1 = halfGraph.toModel()
# print(halfGraph.model.contractionLevel)
# graph = FullGraph(m)
# m1 = graph.toModel()

dissolve = SplitGraph(m)
dissolve.randomize()
dissolve.toModel()
# print(dissolve.model.edgeChannel)
# dissolve.model.show()

for i in range(5):
    dissolve.mutate(1, 0.1)
    dissolve.model.show()
# graph.model.show()
# m = Model(trussParam, simParam)
# m.v0 = m.v0 - m.v0.mean(axis=0)
# m.show()
# # breakpoint()
# mm = MultiMotion(actionSeqs, m)

# mm.animate(0, 10, 10)
