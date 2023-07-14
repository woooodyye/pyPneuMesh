import copy
import pathlib

import numpy as np

from pyPneuMesh.Model import Model


class SplitGraph(object):
    def __init__(self, model, graphSetting):
        self.model = model
        
        self.numChannels = graphSetting['numChannels']#fix this with some value in graphSetting
        #i will figure it out tonight though
        self.groups = graphSetting['groups'] #TODO fix this arbitrary for now 
        

        self.nE = len(model.edgeChannel)
        self.iesSubs =graphSetting['iesSubs'] #maynot exist for non-split graph,
        #should exist though
        #dictionary where each group has a list of iesSub
        # for i in range
        for i in range(len(self.iesSubs)):
            self.iesSubs[i] =  np.array(self.iesSubs[i])

        self.ieAdjLists = []
        self.esSubs = []
        self.incMs = []
        for iesSub in self.iesSubs:
            adjList = self.getAdjList(iesSub)
            self.ieAdjLists.append(adjList)

        self.channels = model.edgeChannel ##TODO this is wrong for now
        self.contractions = model.contractionLevel #TODO fix this also wrong
        

    def channelConnected(self, ic, iGroup):
        # check if channel ic is interconnected
        # return
        esSub = self.esSubs[iGroup]
        incM = self.incMs[iGroup]
        iesSub = self.iesSubs[iGroup]
        nEic = (self.channels == ic).sum()  # number of edges of channel ic
        for ie, e in enumerate(self.iesSubs[iGroup]): #what about this check // think aboutit
            if self.channels[e] == ic:
                break

        queue = [ie]  # ies in the queue
        visited = set()  # ies visited
        
        while queue:
            ie = queue.pop(0)
            visited.add(ie)

            iv0 = esSub[ie, 0]
            iv1 = esSub[ie, 1]
            iesConnected0 = np.where(incM[iv0] == 1)[0]
            iesConnected1 = np.where(incM[iv1] == 1)[0]
            iesConnected = np.concatenate([iesConnected0, iesConnected1])
            for ie in iesConnected:
                if ie not in visited and self.channels[iesSub[ie]] == ic:
                    queue.append(ie)

        return nEic == len(visited)
    def mutateGraph(self):
         # mutate one digit of contractions and one edge channel
        # self.contractions[np.random.choice(len(self.contractions))] = np.random.randint(self.contractionLevels)
        #
        iGroups = np.arange(len(self.ieAdjLists)) # number of groups
        np.random.shuffle(iGroups)
        for iGroup in iGroups:
            if self.mutateGroup(iGroup):
                return True #mutate one of the group, if successful, terminate
        print('mutation failed')
        return False
    
    def mutateGroup(self, iGroup):
        adjList = self.ieAdjLists[iGroup]
        ies = np.arange(len(adjList)) # len of number of edges in that group
        np.random.shuffle(ies) #shuffle to remove bias
        iesSub = self.iesSubs[iGroup]
        for ie in ies: #for each subgraph
            iesConnected = adjList[ie] #adjalencet list for that edge in the group
            
            ieSub = iesSub[ie]

            icOld = self.channels[ieSub] #TODO/ fix this bug
            icsConnected = self.channels[iesSub[iesConnected]].tolist()
            icsConnected = set(icsConnected)
            icsConnected.remove(icOld)
            icsConnected = np.array(list(icsConnected))
            if len(icsConnected):
                np.random.shuffle(icsConnected)
            
            for icNew in icsConnected:
                self.channels[ieSub] = icNew   # change the channel of the edge
                if self.channelConnected(icOld, iGroup):    # if the changed channel is still connected
                    return True     # mutation finished
                else:
                    self.channels[ieSub] = icOld   # revert channel change

    
    def mutateContraction(self, chance):
        #changing contractions doesnt affect the graph anyway
        maskMutation = np.random.rand(len(self.contractions))
        contraction = np.random.randint(
            np.zeros(len(self.contractions)), self.model.NUM_CONTRACTION_LEVEL * np.ones(len(self.contractions)))
        for ie in range(len(self.contractions)):
            if maskMutation[ie] < chance:
                self.contractions[ie] = contraction[ie]
        return 

    def mutate(self, graphMutationChance, contractionMutationChance):
        if np.random.random()< graphMutationChance:
            self.mutateGraph()
        self.mutateContraction(contractionMutationChance)
        self.toModel()

    def saveGraphSetting(self, folderDir, name):
        graphSetting = self.getGraphSetting()

        folderPath = pathlib.Path(folderDir)
        graphSettingPath = folderPath.joinpath("{}.graphsetting".format(name))
        np.save(str(graphSettingPath), graphSetting)


    #Now toModel is part of the graph now
    def toModel(self):
        self.model.contractionLevel = self.contractions
        self.model.edgeChannel = self.channels #contains -1// passive beam in this case I guess
        self.model.edgeChannel = self.model.edgeChannel.astype(np.int64)
        return 
    

    def getAdjList(self, iesSub):
        ivsSub = sorted(set(self.model.e[iesSub].reshape(-1).tolist()))
        esSub = []
        for ieSub in iesSub:
            e = self.model.e[ieSub]
            iv0 = ivsSub.index(e[0])
            iv1 = ivsSub.index(e[1])
            esSub.append([iv0, iv1])
        esSub = np.array(esSub, dtype=int)

        nV = len(ivsSub)
        nE = len(iesSub)
        
        incM = np.zeros([nV, nE])  # vertex-edge incidence matrix
        for ie, e in enumerate(esSub):
            incM[e[0], ie] = 1
            incM[e[1], ie] = 1
        
        ieAdjList = []  # nE x X, each row includes indices of adjacent edges of ie, np.array
        for ie in range(nE):
            iv0 = esSub[ie, 0]
            iv1 = esSub[ie, 1]
            iesConnected0 = np.where(incM[iv0] == 1)[0]
            iesConnected1 = np.where(incM[iv1] == 1)[0]
            iesConnected = np.concatenate([iesConnected0, iesConnected1])
            
            ieAdjList.append(iesConnected)
        self.esSubs.append(esSub) #append esSubs for the group
        self.incMs.append(incM) #append Indicdent vertex matrix for the group
        return ieAdjList

    def randomize(self):
        #randomize each group 
        self.channels = np.ones(self.nE) * -1 #initialize entire graph to -1 
        for i in range(len(self.ieAdjLists)):
            nE = len(self.iesSubs[i]) #select a group of ieSsubs
            ieAdjList = self.ieAdjLists[i] #select adjacent list for a group
            channels = self.randomizeGroup(ieAdjList, nE)
            for ieSub, ie in enumerate(self.iesSubs[i]): #
                #assign channels to the overall channels of the entire graph
                self.channels[ie] = self.numChannels * i + channels[ieSub]
        self.contractions = np.random.randint(0, self.model.NUM_CONTRACTION_LEVEL, self.contractions.shape)
        self.toModel() #Forgot about this
        return
    
    def randomizeGroup(self, ieAdjList, nE):
        #randomize channel to
        channels = np.ones(nE) * -1  # ne, indices of channels of edges

        # randomly choose numChannel edges and assign channels
        dice = np.arange(nE)
        np.random.shuffle(dice)
        ies = dice[:self.numChannels]
        for ic, ie in enumerate(ies):
            channels[ie] = ic

        # grow channels to fill the entire graph
        while (channels == -1).any():
            iesToGrow = []
            for ie in range(nE):
                iesConnected = ieAdjList[ie]
                if (channels[iesConnected] == -1).any() and channels[ie]!=-1: #TODO newly added
                    iesToGrow.append(ie)

            ie = np.random.choice(iesToGrow)
            iesConnected = ieAdjList[ie]
            np.random.shuffle(iesConnected)
            for ieConnected in iesConnected:
                if channels[ieConnected] == -1:
                    channels[ieConnected] = channels[ie]
        return channels
        
        
    def cross(self, graph, chance):
        maskMutation = np.random.rand(len(self.contractions))
        for ie in range(len(self.contractions)):
            if maskMutation[ie] < chance:
                tmp = self.contractions[ie]
                self.contractions[ie] = graph.contractions[ie]
                graph.contractions[ie] = tmp
        #does this ever get applies even, only mutatae the graph, not the model
        # graph.toModel()
        #TODO do this for splitgraph
        self.toModel()
        graph.toModel()

    def getGraphSetting(self):
        graphSetting = {
            'symmetric': False,
            'dissolve': True, 
            'iesSubs' : self.iesSubs,
            'numChannels' : self.numChannels,
            'groups' : self.groups
        }
        return copy.deepcopy(graphSetting)
