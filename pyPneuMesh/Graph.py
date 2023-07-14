import pathlib
import copy
import numpy as np

from pyPneuMesh.FullGraph import FullGraph
from pyPneuMesh.HalfGraph import HalfGraph
from pyPneuMesh.SplitGraph import SplitGraph


class Graph(object):

    def __new__(cls, graphSetting, model):
        symmetric = graphSetting['symmetric']
        # dissolve = graphSetting.get('dissolve', False) #Default graph to be non dissolve
        dissolve = graphSetting['dissolve']
        graphSetting = copy.deepcopy(graphSetting)
        if symmetric:
            return HalfGraph(model, graphSetting)
        if dissolve:
            return SplitGraph(model, graphSetting)
        else:
            return FullGraph(model)

    def __init__(self, graphSetting, model):
        self.graphSetting = graphSetting.copy()
        self.model = model
        pass

    def saveGraphSetting(self, folderDir, name):
        pass

    def getGraphSetting(self):
        pass

    def mutate(self, graphMutationChance, contractionMutationChance):
        pass

    def cross(self, graph, chance):
        pass

    def randomize(self):
        pass

    def saveGraphSetting(self):
        pass
