import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyPneuMesh.utils import readNpy



# sns.set_theme(style="darkgrid")
# plt.gca().set_facecolor((0.95,0.95,0.95, 1.0))

#2 missing for now
# L = ["2", "8", "16", "32", "64"]
np_data = readNpy("scripts/trainLobster/output/2023-02-02_16-23-13/ElitePool_139.gacheckpoint.npy")

genes = np_data['genePoolMOODict']
score_move_x = []
score_move_y = []

score_min_x = [] # distance traveled
score_min_y = [] #min energy
for gene in genes:
    score = gene['score']
    score_move_x.append(score[0])
    score_min_x.append(score[1])
    score_min_y.append(score[2])
    # Create a line plot of performance over iterations
    score_move_y.append(0)


fig, ax = plt.subplots()

#set the tick params
ax.tick_params(direction="in")
plt.scatter(score_move_x,score_move_y, label= "move forward only", color = "red")  # Add a label
plt.scatter(score_min_x,score_min_y, label= "move forward and minimize energy", color = "blue")  # Add a label

# Add the shaded variance
# plt.fill_between(iterations, lower, upper, color='b', alpha=.1)


plt.title('Gene Pool')
plt.xlabel('Distance Moved', fontname ='PT Sans Narrow')
plt.ylabel('Energy, $E / d$', fontname ='PT Sans Narrow')
plt.ylim((-500, 10))

plt.legend(loc='lower right', frameon = False)  # Show the legend

plt.savefig("test_lobster.png")