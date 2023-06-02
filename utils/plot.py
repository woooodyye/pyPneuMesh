# can integrate io command latter, but right now just do simple things

# from pyPneuMesh.GA import GA
import matplotlib.pyplot as plt


# elitePools = {2:'scripts/trainTable_2/output/2023-03-16_17-13-07/',
#               8:'scripts/trainTable_8/output/2023-03-16_17-16-25/', 
#               16: 'scripts/trainTable_16/output/2023-02-23_19-02-40/', 
#               32: 'scripts/trainTable_32/output/2023-04-06_04-05-10/',
#              64 : 'scripts/trainTable_64/output/2023-03-22_18-01-24/', 
# }

elitePools = {2:'scripts/trainTable_2/output/2023-03-16_17-13-07/',
              8:'scripts/trainTable_8/output/2023-03-16_17-16-25/', 
              16: 'scripts/trainTable_16/output/2023-04-26_20-37-43/', 
              32: 'scripts/trainTable_32/output/2023-04-06_04-05-10/',
             64 : 'scripts/trainTable_64/output/2023-03-22_18-01-24/', 
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
    for i, value in enumerate(results):
        if value != []:
            x.append(i)
            y.append(value[1])
    
    plt.plot(x,y, '-o', label = str(j))


plt.legend()
plt.xlabel("iterations")
plt.ylabel("max Distance")
plt.title("max Distance vs Iterations for Table with multiple channels")
plt.savefig("trainTable_alliterations_maxDistance.png")

    # plt.plot(index, maxDistance)
    # # plt.axis([0, 10, 3.8, 5])
    # plt.savefig("test.png")
