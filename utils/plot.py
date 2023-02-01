# can integrate io command latter, but right now just do simple things
from parse import parse, search, findall

from pyPneuMesh.GA import GA
import matplotlib.pyplot as plt

fileDir = 'scripts/trainLobster/output/2023-01-26_15-50-46/'
ga = GA(GACheckpointDir=fileDir + 'ElitePool_8.gacheckpoint.npy')
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
    print(vals)
    for val in vals:
        try:
            result.append(float(val))
        except:
            continue
    results.append(result)
# format = "ElitePool:{i} {mean1} / {max1}{mean2} / {max2}{mean3} / {max3}"
# print([parse(format, elitePool).named for elitePool in ElitePools])

index = []
maxDistance = []
for i, value in enumerate(results):
    if value != []:
        index.append(i)
        maxDistance.append(value[1])

print(index)
print(maxDistance)
plt.plot(index, maxDistance)
# plt.axis([0, 10, 3.8, 5])
plt.show()
