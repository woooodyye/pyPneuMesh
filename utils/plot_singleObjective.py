import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# sns.set_theme(style="darkgrid")
plt.gca().set_facecolor((0.95,0.95,0.95, 1.0))
fig, ax = plt.subplots()

#set the tick params
ax.tick_params(direction="in")


np_data = np.load("utils/plot_data/single_objective_allchannels.npy")
L = ["2", "8", "16", "32", "64"]
for i in range(5):
    iterations = np_data[0][0] #1 x n
    performances = np_data[1][i]

    # Create a line plot of performance over iterations
    plt.plot(iterations, performances, label= L[i])  # Add a label
    # Add the shaded variance

plt.title('Single Objective Training performance across channels')
plt.xlabel('Generation')
plt.ylabel('Max Distance Traveld')

plt.legend(loc='lower right', frameon = False)  # Show the legend

plt.savefig("singleobjective_table_all.png")