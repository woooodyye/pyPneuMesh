# can integrate io command latter, but right now just do simple things

# from pyPneuMesh.GA import GA
import matplotlib.pyplot as plt
import numpy as np

elitePools = {1 : 'scripts/trainTable_16/output/2023-02-23_19-02-40/', 
              2: 'scripts/trainTable_16/output/2023-04-26_05-51-48/', 
              3: 'scripts/trainTable_16/output/2023-04-26_20-37-43/'
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
    for i, value in enumerate(results):
        if value != []:
            x.append(i)
            y.append(value[1])
            a = np.linspace(1, 1000, 1000)
            b = np.interp(a, x, y)
    xx.append(a)
    yy.append(b)

x_mean = np.mean(xx, axis= 0)
y_mean = np.mean(yy, axis =0)
plt.plot(x_mean,y_mean, '-o', label = str(16))

# plt.legend()
plt.xlabel("iterations")
plt.ylabel("max Distance")
plt.title("max Distance vs Iterations for Table with 8 channels multiple runs")
plt.savefig("trainTable_8_mult_max_mean_val.png")

# plt.legend()
# plt.xlabel("iterations")
# plt.ylabel("mean Distance")
# plt.title("mean Distance vs Iterations for Table with 16 channels multiple runs")
# plt.savefig("trainTable_16_mult_mean.png")

    # plt.plot(index, maxDistance)
    # # plt.axis([0, 10, 3.8, 5])
    # plt.savefig("test.png")
