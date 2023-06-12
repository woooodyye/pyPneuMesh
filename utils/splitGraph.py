import numpy as np
from utils.graph import Graph

#everything becomes a dictionary ? 
#large memory usage // easy to implement and understand
#using exisiting structure
#assuming each graph has 2 channels 
class SplitGraph(object):
    def __init__(self, numChannels:int, ivsSubs : dict, iesSubs: dict, esSubs : dict,contractionLevels:int, 
                numGraphs: int):
        #try to initialize and split up graph into subcategory.
        self.numChannels = int(numChannels)
        self.numGraphs = numGraphs
        self.contractionLevels = contractionLevels
        self.graphs = {}
        self.iesSubs = iesSubs

        for i in range(self.numGraphs):
            iesSub = iesSubs[i]
            ivsSub = ivsSubs[i]
            esSub = esSubs[i]
            self.graphs[i] = Graph(numChannels=self.numChannels, ivsSub=ivsSub, iesSub=iesSub, esSub=esSub, contractionLevels=self.contractionLevels)
    def mutate(self):
        #mutate each graph
        for i in range(self.numGraphs):
            self.graphs[i].mutate()