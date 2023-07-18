import os
import pathlib
import time

rootPath = os.path.split(os.path.realpath(__file__))[0]
rootPath = os.path.split(rootPath)[0]

from pyPneuMesh.utils import getDefaultValue, getLength
from build.model import Model as CModel

import numpy as np
import copy


class Model(object):

    def __init__(self, trussParam, simParam):
        trussParam = copy.deepcopy(trussParam)
        simParam = copy.deepcopy(simParam)

        # load from trussParam
        self.v0 = trussParam['v0']
        self.e = trussParam['e']
        
        #NewLobster System 
        eSubs =[6, 8, 28, 34, 35, 36, 41, 46, 48, 49, 50, 51, 52, 64, 66, 67, 72, 74, 77, 85, 100, 101, 106, 107, 108, 109, 111, 116, 120, 121, 126, 131, 133,
2, 4, 7, 10, 14, 21, 24, 29, 32, 33, 37, 40, 42, 43, 47, 62, 65, 71, 78, 79, 83, 95, 96, 98, 99, 104, 105, 110, 115, 117, 118, 124, 127,
5, 9, 13, 15, 16, 17, 18, 20, 22, 23, 25, 27, 30, 31, 53, 54, 55, 57, 58, 59, 60, 61, 63, 68, 69, 73, 75, 76, 81, 82, 84, 86, 87, 88, 89, 91, 92, 94, 97, 112, 113, 114, 122, 123, 128, 129, 132, 134, 135, 136]# this is some abitrary value fof lobster testing for now
        self.eSubs = np.array(eSubs, dtype = int)
        self.edgeChannel = getDefaultValue(trussParam, 'edgeChannel',
                                           np.zeros(len(self.e), dtype=int)  # [0, 0, ...]
                                           )
        self.edgeActive = getDefaultValue(trussParam, 'edgeActive',
                                          np.ones(len(self.e), dtype=bool)  # [True, True, ...]
                                          )
        self.vertexFixed = getDefaultValue(trussParam, 'vertexFixed',
                                           np.zeros(len(self.e), dtype=bool)  # [False, False, ...]
                                           )
        self.contractionLevel = getDefaultValue(trussParam, 'contractionLevel',
                                                np.zeros(len(self.e), dtype=int)  # [False, False, ...]
                                                )
        self.CONTRACTION_SPEED = getDefaultValue(trussParam, 'CONTRACTION_SPEED',
                                                 0
                                                 )
        self.NUM_CONTRACTION_LEVEL = getDefaultValue(trussParam, 'NUM_CONTRACTION_LEVEL',
                                                     0
                                                     )
        self.CONTRACTION_PER_LEVEL = getDefaultValue(trussParam, 'CONTRACTION_PER_LEVEL',
                                                     0
                                                     )
        self.MAX_ACTIVE_BEAM_LENGTH = getDefaultValue(trussParam, 'MAX_ACTIVE_BEAM_LENGTH',
                                                      np.max(getLength(self.v0, self.e))
                                                      )
        self.ACTION_TIME = getDefaultValue(trussParam, 'ACTION_TIME',
                                           5.0
                                           )
        # load from simParam
        self.k = float(simParam['k'])
        self.h = float(simParam['h'])
        self.gravity = float(simParam['gravity'])
        self.damping = float(simParam['damping'])
        self.friction = float(simParam['friction'])

        # calculate maxLengths
        maxLengths = getLength(self.v0, self.e)
        maxLengths[self.edgeActive] = self.MAX_ACTIVE_BEAM_LENGTH
        self.maxLengths = maxLengths

    def save(self, folderDir, name):
        trussParam = self.getTrussParam()
        simParam = self.getSimParam()

        folderPath = pathlib.Path(folderDir)
        trussParamPath = folderPath.joinpath("{}.trussparam".format(name))
        simParamPath = folderPath.joinpath("{}.simparam".format(name))

        np.save(
            str(trussParamPath),
            trussParam
        )
        np.save(
            str(simParamPath),
            simParam
        )

    def getTrussParam(self):
        trussParam = {
            'v0': self.v0,
            'e': self.e,
            'edgeChannel': self.edgeChannel,
            'edgeActive': self.edgeActive,
            'vertexFixed': self.vertexFixed,
            'contractionLevel': self.contractionLevel,
            'CONTRACTION_SPEED': self.CONTRACTION_SPEED,
            'NUM_CONTRACTION_LEVEL': self.NUM_CONTRACTION_LEVEL,
            'CONTRACTION_PER_LEVEL': self.CONTRACTION_PER_LEVEL,
            'MAX_ACTIVE_BEAM_LENGTH': self.MAX_ACTIVE_BEAM_LENGTH,
            'ACTION_TIME': self.ACTION_TIME
        }
        return copy.deepcopy(trussParam)

    def getSimParam(self):
        simParam = {
            'k': self.k,
            'h': self.h,
            'damping': self.damping,
            'gravity': self.gravity,
            'friction': self.friction
        }
        return copy.deepcopy(simParam)

    def actionSeq2timeNLength(self, actionSeq, dissolve):
        # actionSeq: [numActions, numChannel]
        # Change this inresponse to dissolve, separate the length and contractions
        # into different groups now 
        #TODO Make this better formatted, hardcode for now to test things 

        time = 0
        times = [time]
        # lengths = [self.maxLengths.copy()]  # target lengths
        #Passive Beam channel is -1 #dissolvable beam same idea? 
        #Double check this later
        eSubs = np.arange(0,len(self.maxLengths))
        if dissolve:
            eSubs = self.eSubs

        lengths = [self.maxLengths.copy()[eSubs]]

        for iAction in range(len(actionSeq)):
            action = actionSeq[iAction]
            length = self.maxLengths.copy()[eSubs] #SubLengths Ones with edges in the group

            contraction = np.zeros(len(eSubs))
            for iChannel in range(self.getNumChannel()):
                if not action[iChannel]:  # not contracting
                    continue  # contraction does not change
                else:
                    boolThisChannel = self.edgeChannel == iChannel
                    boolThisChannel = boolThisChannel[eSubs] #TODO double checkthis newly added
                    contraction[boolThisChannel] = self.CONTRACTION_PER_LEVEL * \
                                                   self.contractionLevel[eSubs][boolThisChannel] * \
                                                   action[iChannel]

            if dissolve:
                length -= contraction #All of them should be active
            else:
                length[self.edgeActive] -= contraction[self.edgeActive] #How to deal with activeEdges

            time += self.ACTION_TIME
            times.append(time)
            lengths.append(length)

        times = np.array(times, dtype=np.float64)
        lengths = np.array(lengths, dtype=np.float64)

        return times, lengths

    def step(self, numSteps=1, times=None, lengths=None, dissolve = False):
        if times is None:
            times = np.array([0.0], dtype=np.float64)
            lengths = np.array([getLength(self.v0, self.e)], dtype=np.float64).reshape(1, len(self.e))

        assert (times.ndim == 1 and times.dtype == np.float64)
        assert (lengths.ndim == 2 and lengths.shape[0] == times.shape[0] and lengths.dtype == np.float64)


        #pusdo
        # e nonDissolvable .. objective ... 
        e = self.e
        if dissolve:
            e = self.e[self.eSubs]
        cModel = CModel(self.k, self.h, self.gravity, self.damping, self.friction, self.v0, e,
                        self.CONTRACTION_SPEED)
        vs, vEnergys = cModel.step(times, lengths, numSteps)
        vs = vs.reshape((numSteps + 1), len(self.v0), 3)
        vEnergys = vEnergys.reshape((numSteps + 1), len(e))
        return vs, vEnergys

    def show(self, v=None):
        import polyscope as ps
        try:
            ps.init()
        except:
            pass
        ps.set_up_dir('z_up')

        if v is None:
            v = self.v0

        for ic in list(np.arange(self.getNumChannel())):
            ies = np.arange(len(self.e))[self.edgeChannel == ic]
            es = self.e[ies]
            cs = ps.register_curve_network(str(ic), v, es)

        ps.register_point_cloud('15', v[15:16])
        ps.show()

    def animate(self, vs, speed=1.0, singleColor=False):
        import polyscope as ps
        try:
            ps.init()
        except:
            pass
        ps.set_up_dir('z_up')

        if not singleColor:
            css = []
            for ic in list(np.arange(self.getNumChannel())):
                ies = np.arange(len(self.e))[self.edgeChannel == ic]
                es = self.e[ies]
                cs = ps.register_curve_network(str(ic), vs[0], es)
                css.append(cs)

            t0 = time.time()

            def callback():
                t = (time.time() - t0) * speed
                iStep = int(t / self.h)
                try:
                    v = vs[iStep]
                    for cs in css:
                        cs.update_node_positions(v)
                except:
                    pass

        else:
            cs = ps.register_curve_network('truss', vs[0], self.e)

            t0 = time.time()

            def callback():
                t = (time.time() - t0) * speed
                iStep = int(t / self.h)
                try:
                    v = vs[iStep]
                    cs.update_node_positions(v)
                except:
                    pass
        ps.register_curve_network('y axis', np.array([[0, 0, 0], [0, 1, 0]]), np.array([[0, 1]]))
        ps.register_curve_network('x axis', np.array([[0, 0, 0], [1, 0, 0]]), np.array([[0, 1]]))
        ps.set_user_callback(callback)
        ps.show()

    def getNumChannel(self):
        return max(self.edgeChannel) + 1
