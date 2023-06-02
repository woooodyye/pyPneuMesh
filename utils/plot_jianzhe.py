import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# sns.set_theme(style="darkgrid")
plt.gca().set_facecolor((0.95,0.95,0.95, 1.0))

#2 missing for now
# L = ["2", "8", "16", "32", "64"]
L = [ "2", "8", "16", "32", "64"]

fig, ax = plt.subplots()

#set the tick params
ax.tick_params(direction="in")

for i in L:
    np_data = np.load("utils/plot_data/" + i + ".npy")

    iterations = np_data[0][0] #1 x n
    performances = np.mean(np_data[1], axis = 0) #3x  n too 
    variances = np.var(np_data[1], axis = 0)

    #calculate the shaded variances .. 
    upper = performances + variances
    lower = performances - variances


    # Create a line plot of performance over iterations
    plt.plot(iterations, performances, label= i)  # Add a label

    # Add the shaded variance
    plt.fill_between(iterations, lower, upper, color='b', alpha=.1)

    plt.title('Training Performance', fontname ='PT Sans Narrow')
    plt.xlabel('Generation',fontname ='PT Sans Narrow')
    plt.ylabel('Pareto Score', fontname ='PT Sans Narrow')

    plt.legend(loc='lower right', frameon = False)  # Show the legend

plt.savefig("tableAllChannels.png")