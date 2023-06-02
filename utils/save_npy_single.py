# can integrate io command latter, but right now just do simple things

# from pyPneuMesh.GA import GA
import matplotlib.pyplot as plt
import numpy as np
import pathlib

elitePools = {1 : 'scripts/trainTable_2_move/output/2023-05-18_16-15-04/', 
              2: 'scripts/trainTable_8_move/output/2023-05-16_23-21-32/', 
              3: 'scripts/trainTable_16_move/output/2023-05-16_23-18-07/',
              4: 'scripts/trainTable_32_move/output/2023-04-27_15-09-36/',
              5: 'scripts/trainTable_64_move/output/2023-04-27_05-51-56/'
}

# elitePools = {1 : 'scripts/trainTable_8/output/2023-03-16_17-16-25/', 
#               2: 'scripts/trainTable_8/output/2023-04-26_05-52-20/', 
#               3: 'scripts/trainTable_8/output/2023-04-26_05-53-40/'
# }

xx = []
yy = []
for j, fileDir in elitePools.items():

    infile = fileDir + 'log.txt'

    with open(infile) as f:
        f = f.readlines()

    ElitePools = []

    for line in f:
        # dumb way, can also just try to step through lines knowning the number of genes
        # and the file format, le'ts see if this is too slow
        if 'ElitePool' in line and 'Training' not in line:
            ElitePools.append(line)

    # print(ElitePools)
    # i = 0
    results = []
    for elitePool in ElitePools:
        # vals = elitePool.split(" ")
        # I can just install the parse library too
        # result = elitePool.parse()
        vals = elitePool.split()
        result = []
        for val in vals:
            try:
                result.append(float(val))
            except:
                continue
        results.append(result)
    # format = "ElitePool:{i} {mean1} / {max1}{mean2} / {max2}{mean3} / {max3}"
    # print([parse(format, elitePool).named for elitePool in ElitePools])

    x = []
    y = []
    #need to cut results length to be consistent
    length = 500
    results = results[0:length]
    #500 iterations
    for i, value in enumerate(results):
        if value != []:
            x.append(i)
            y.append(value[1])
            #max value here, only single objective training
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
name = "single_objective_allchannels"
objectivesPath = folderPath.joinpath("{}".format(name))
np.save(str(objectivesPath), np_output)


