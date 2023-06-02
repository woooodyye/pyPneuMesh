# can integrate io command latter, but right now just do simple things

# from pyPneuMesh.GA import GA
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from pyPneuMesh.utils import readNpy

from pymoo.indicators.hv import HV



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



# elitePools = {1 : 'scripts/trainTable_16/output/2023-02-23_19-02-40/', 
#               2: 'scripts/trainTable_16/output/2023-04-26_05-51-48/', 
#               3: 'scripts/trainTable_16/output/2023-04-26_20-37-43/'
# }


#8 channels
# elitePools = {1 : 'scripts/trainTable_8/output/2023-03-16_17-16-25/', 
#               2: 'scripts/trainTable_8/output/2023-04-26_05-52-20/', 
#               3: 'scripts/trainTable_8/output/2023-04-26_05-53-40/'
# }

elitePools = {1 : 'scripts/trainTable_32/output/2023-04-06_04-05-10/', 
              2: 'scripts/trainTable_32/output/2023-05-12_21-16-26/', 
              3: 'scripts/trainTable_32/output/2023-05-15_17-07-16/'
}
# elitePools = {1 : 'scripts/trainTable_64/output/2023-03-22_18-01-24/', 
#               2: 'scripts/trainTable_64/output/2023-05-10_17-48-32/', 
#               3: 'scripts/trainTable_64/output/2023-05-11_17-31-50/'
# }

# elitePools = {1 : 'scripts/trainTable_2/output/2023-03-16_17-13-07/', 
#               2: 'scripts/trainTable_2/output/2023-05-18_18-32-23/', 
#               3: 'scripts/trainTable_2/output/2023-05-18_19-30-10/'
# }

xx = []
yy = []

ref_point = np.array([10, 1, 1, 1])
# ref_point = np.array([10])
xx = []
yy = []
for j, elitePoolsDir in elitePools.items():

    ind = HV(ref_point=ref_point)

    x = []
    y = []

    #need to modify this. Change to 1000 for consistency later.
    length = 1000
    
    for i in range(0,length,1):
        elitePoolName=  elitePoolsDir + "ElitePool_" + str(i) + '.gacheckpoint.npy'

        try:
            elitePool =readNpy(elitePoolName)
        except:
            break
        

        elitePoolMooDict =elitePool['elitePoolMOODict']
        n = len(elitePoolMooDict)
        if(n == 0):
            continue

        m = len(elitePoolMooDict[0]['score']) #hard coded for now
        
        arr = np.zeros((n,m))
        for num in range(len(elitePoolMooDict)):
            arr[num] = elitePoolMooDict[num]['score']

        Rs,CDs = getRCD(arr)
        scores = arr[Rs == 0] 
        m = len(scores)
        scores = scores[:, ]

        x.append(i)
        y.append(ind(-scores))
    
        #find linSpace of 1000 value 
    a = np.linspace(1, length, length)
    b = np.interp(a, x, y)
    xx.append(a)
    yy.append(b)

x_np = np.asarray(xx, dtype=np.float64)
y_np = np.asarray(yy, dtype=np.float64)

x_np = np.reshape(x_np, (1, x_np.shape[0],x_np.shape[1]))
y_np = np.reshape(y_np, (1, y_np.shape[0],y_np.shape[1]))
print(x_np.shape)
print(y_np.shape)

np_output = np.concatenate((x_np,y_np))

print(np_output.shape)


folderDir = "utils/plot_data/"
folderPath = pathlib.Path(folderDir)
name = "32"
objectivesPath = folderPath.joinpath("{}".format(name))
np.save(str(objectivesPath), np_output)


