# can integrate io command latter, but right now just do simple things

# from pyPneuMesh.GA import GA
import matplotlib.pyplot as plt


# elitePools = {2:'scripts/trainTable_2/output/2023-03-16_17-13-07/',
#               8:'scripts/trainTable_8/output/2023-03-16_17-16-25/', 
#               16: 'scripts/trainTable_16/output/2023-02-23_19-02-40/', 
#               32: 'scripts/trainTable_32/output/2023-04-06_04-05-10/',
#              64 : 'scripts/trainTable_64/output/2023-03-22_18-01-24/', 
# }

elitePools = {2:'scripts/trainLobster/output/2023-02-02_16-23-13/'
}

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
    yBoth = []
    for i, value in enumerate(results):
        if value != []:
            x.append(i)
            y.append(value[1])
            yBoth.append(value[3])


fig, ax = plt.subplots()
#set the tick params
ax.tick_params(direction="in")

plt.plot(x,y, label= "move forward only", color = "red")  # Add a label
plt.plot(x,yBoth, label= "move forward and minimize energy", color = "blue")  # Add a label

# Add the shaded variance
# plt.fill_between(iterations, lower, upper, color='b', alpha=.1)
titleName = "Iteration vs Max Distance Traveled"
xLabelName = "Iterations"
yLabelName = "Max Distance Travelled"
figureName = "lobsterLog.png"
plt.title(titleName)
plt.xlabel(xLabelName, fontname ='PT Sans Narrow')
plt.ylabel(yLabelName, fontname ='PT Sans Narrow')
# plt.ylim((-500, 10))
plt.legend(loc='lower right', frameon = False)  # Show the legend
plt.savefig(figureName)