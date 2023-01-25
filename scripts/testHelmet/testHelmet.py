from pyPneuMesh.utils import readNpy
from pyPneuMesh.Model import Model
from pyPneuMesh.MultiMotion import MultiMotion

trussParam = readNpy('scripts/testHelmet/data/helmet.trussparam.npy')
simParam = readNpy('scripts/testHelmet/data/helmet.simparam.npy')
actionSeqs = readNpy('scripts/testHelmet/data/helmet.actionseqs.npy')

m = Model(trussParam, simParam)
mm = MultiMotion(actionSeqs, m)

#

mm.animate(0, 10, 10)

# trussParam = readNpy('scripts/testLobster/lobster/lobster.trussparam.npy')
# simParam = readNpy('scripts/testLobster/lobster/lobster.simparam.npy')
# actionSeqs = readNpy('scripts/testLobster/lobster/Lobster_0.actionseqs.npy')
#
# m = Model(trussParam, simParam)
# mm = MultiMotion(actionSeqs, m)
#
# mm.animate(0, 10, 10)
