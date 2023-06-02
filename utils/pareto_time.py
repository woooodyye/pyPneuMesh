import numpy as np
from pyPneuMesh.utils import readNpy
import matplotlib.pyplot as plt

from pymoo.indicators.hv import HV


#Open all the elitePool
#For each ElitePool, get the pareto front 
#Calculate HV

def getRCD(ratings: np.ndarray) -> (np.ndarray, np.ndarray):
        if len(ratings) == 0:
            return np.array([]), np.array([])
        # get R
        ratings = ratings.reshape(len(ratings), -1)
        ratingsCol = ratings.reshape([ratings.shape[0], 1, ratings.shape[1]])
        ratingsRow = ratings.reshape([1, ratings.shape[0], ratings.shape[1]])
        dominatedMatrix = (ratingsCol <= ratingsRow).all(2) * (ratingsCol < ratingsRow).any(2)
        
        Rs = np.ones(len(ratings), dtype=int) * -1
        R = 0
        while -1 in Rs:
            nonDominated = (~dominatedMatrix[:, np.arange(len(Rs))[Rs == -1]]).all(1) * (Rs == -1)
            Rs[nonDominated] = R
            R += 1
            
        # get CD
        CDs = np.zeros(len(ratings))
        R = 0
        while R in Rs:
            ratingsSurface = ratings[Rs == R]
            CDMatrix = np.zeros_like(ratingsSurface, dtype=np.float)
            sortedIds = np.argsort(ratingsSurface, axis=0)
            CDMatrix[sortedIds[0], np.arange(len(ratingsSurface[0]))] = np.inf
            CDMatrix[sortedIds[-1], np.arange(len(ratingsSurface[0]))] = np.inf
            ids0 = sortedIds[:-1, :]
            ids1 = sortedIds[1:, :]
            distances = ratingsSurface[ids1, np.arange(len(ratingsSurface[0]))] - ratingsSurface[ids0, np.arange(len(ratingsSurface[0]))]
            
            if ((ratingsSurface.max(0) - ratingsSurface.min(0)) > 0).all():
                CDMatrix[sortedIds[1:-1, :], np.arange(len(ratingsSurface[0]))] = \
                    (distances[1:] + distances[:-1]) / (ratingsSurface.max(0) - ratingsSurface.min(0))
            else:
                CDMatrix[sortedIds[1:-1, :], np.arange(len(ratingsSurface[0]))] = np.inf
            CDsSurface = CDMatrix.mean(1)
            CDs[Rs == R] = CDsSurface
            R += 1
        
        
        return Rs, CDs


# elitePools = {2:'scripts/trainTable_2/output/2023-03-16_17-13-07/',
#               8:'scripts/trainTable_8/output/2023-03-16_17-16-25/', 
#               16: 'scripts/trainTable_16/output/2023-02-23_19-02-40/', 
#               32: 'scripts/trainTable_32/output/2023-02-26_19-39-41/',
#              64 : 'scripts/trainTable_64/output/2023-02-26_19-51-44/', 
# }
# elitePools = {
#               16: 'scripts/trainTable_16/output/2023-02-23_19-02-40/', 
#               32: 'scripts/trainTable_32/output/2023-02-26_19-39-41/',
#              64 : 'scripts/trainTable_64/output/2023-03-22_18-01-24/', 
# }
elitePools = {
               32: 'scripts/trainTable_32/output/2023-04-06_04-05-10/',
               64 : 'scripts/trainTable_64/output/2023-03-22_18-01-24/'
              }


# ref_point = np.array([0, 2000])
# ref_point = np.array([10, 1, 0.7, 1])
# ref_point = np.array([10, 1, 1, 1])


ref_point = np.array([10, 10, 10, 10])
# ref_point = np.array([10])

for j, elitePoolsDir in elitePools.items():

    ind = HV(ref_point=ref_point)

    x = []
    y = []
    t = 0
    for i in range(500):
        elitePoolName=  elitePoolsDir + "ElitePool_" + str(i) + '.gacheckpoint.npy'
        elitePool =readNpy(elitePoolName)
        
        elitePoolMooDict =elitePool['elitePoolMOODict']
        n = len(elitePoolMooDict)
        # print(n)
        #NO ElitePool
        if(n == 0):
            continue
        
        m = len(elitePoolMooDict[0]['score']) #hard coded for now
        
        arr = np.zeros((n,m))
        for num in range(len(elitePoolMooDict)):
            arr[num] = elitePoolMooDict[num]['score']

        Rs,CDs = getRCD(arr)
        scores = arr[Rs == 0] 
        m = len(scores)
        # scores = scores[:, [0]]
        t = elitePool['secondsPassed'] / 60
        x.append(t) #record minutes?
        y.append(ind(-scores))
    plt.plot(x, y, '-o', label = str(j))

plt.legend()
plt.savefig("trainTable_32vs64timed.png")


