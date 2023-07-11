from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.MultiMotion import MultiMotion

# trussParam = readNpy('scripts/testLobster/lobster/lobster.trussparam.npy')
# simParam = readNpy('scripts/testLobster/lobster/lobster.simparam.npy')
# actionSeqs = readNpy('scripts/testLobster/lobster/Lobster_0.actionseqs.npy')
#
# m = Model(trussParam, simParam)
# mm = MultiMotion(actionSeqs, m)
# m.v0 = m.v0 - m.v0.mean(axis=0)
# m.show()
# breakpoint()
#
# mm.animate(0, 10, 10)
# breakpoint()
# trussParam = readNpy('scripts/testLobster/lobster/lobster_grabgo.trussparam.npy')
# simParam = readNpy('scripts/testLobster/lobster/lobster.simparam.npy')
# actionSeqs = readNpy('scripts/testLobster/lobster/lobster_grabgo.actionseqs.npy')
#
# m = Model(trussParam, simParam)
# m.v0 = m.v0 - m.v0.mean(axis=0)
# m.show()
# breakpoint()
# mm = MultiMotion(actionSeqs, m)
#
# mm.animate(0, 10, 10)

trussParam = readNpy('scripts/testLobster/lobster/lobster_walk.trussparam.npy')
simParam = readNpy('scripts/testLobster/lobster/lobster.simparam.npy')
actionSeqs = readNpy('scripts/testLobster/lobster/lobster_walk.actionseqs.npy')

m = Model(trussParam, simParam)
m.v0 = m.v0 - m.v0.mean(axis=0)
m.show()
# breakpoint()
mm = MultiMotion(actionSeqs, m)

mm.animate(0, 10, 10)
