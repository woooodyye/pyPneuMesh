from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.MultiMotion import MultiMotion
from pyPneuMesh.MultiObjective import MultiObjective
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO

trussParam = readNpy('scripts/testMOO/table/table.trussparam.npy')
simParam = readNpy('scripts/testMOO/table/table.simparam.npy')
actionSeqs = readNpy('scripts/testMOO/table/table.actionseqs.npy')
objectives = readNpy('scripts/testMOO/table/table.objectives.npy')
graphSetting = readNpy('scripts/testMOO/table/table.graphsetting.npy')

m = Model(trussParam, simParam)
mm = MultiMotion(actionSeqs, m)
mo = MultiObjective(objectives, mm)
g = Graph(graphSetting, m)
moo = MOO(multiObjective=mo, graph=g, randomize=False)

print(moo.evaluate())
moo.mutate(contractionMutationChance=0.0, actionMutationChance=0.0, graphMutationChance= 0.0)
print(moo.evaluate())
moo.model.show()
g.randomize()
print(moo.evaluate())
moo.randomize()
print(moo.evaluate())
moo.mutate(contractionMutationChance=0.0, actionMutationChance=0.1, graphMutationChance= 1)
moo.model.show()
print(moo.evaluate())
moo.mutate(contractionMutationChance=0.1, actionMutationChance=0.0)
print(moo.evaluate())

print(' ')

mooDict = readMooDict('examples/testMOO/table')

moo = MOO(mooDict=mooDict, randomize=False)

print(moo.evaluate())
moo.mutate(contractionMutationChance=0.0, actionMutationChance=0.0)
print(moo.evaluate())
moo.graph.randomize()
print(moo.evaluate())
moo.randomize()
print(moo.evaluate())
moo.mutate(contractionMutationChance=0.0, actionMutationChance=0.1)
print(moo.evaluate())
moo.mutate(contractionMutationChance=0.1, actionMutationChance=0.0)
print(moo.evaluate())





