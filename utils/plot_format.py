import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


titleName= "x"  
xLabelName = "x"
yLabelName = "y"
figureName ="fig.png"

# sns.set_theme(style="darkgrid")
plt.gca().set_facecolor((0.95,0.95,0.95, 1.0))

#2 missing for now
# L = ["2", "8", "16", "32", "64"]

L = [0, 1, 2, 3, 4, 5]
score_move_x = []
score_move_y = []

score_min_x = [] # distance traveled
score_min_y = [] #min energy
for i in L:
    score = i 
    score_move_x.append(i+1)
    score_min_x.append(i* 2)
    score_min_y.append(i * i)
    # Create a line plot of performance over iterations
    score_move_y.append(0)

fig, ax = plt.subplots()

#set the tick params
ax.tick_params(direction="in")


plt.scatter(score_move_x,score_move_y, label= "move forward only", color = "red")  # Add a label
plt.scatter(score_min_x,score_min_y, label= "move forward and minimize energy", color = "blue")  # Add a label

# Add the shaded variance
# plt.fill_between(iterations, lower, upper, color='b', alpha=.1)


plt.title(titleName)
plt.xlabel(xLabelName, fontname ='PT Sans Narrow')
plt.ylabel(yLabelName, fontname ='PT Sans Narrow')
plt.ylim((-500, 10))
plt.legend(loc='lower right', frameon = False)  # Show the legend
plt.savefig(figureName)