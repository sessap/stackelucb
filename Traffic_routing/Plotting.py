
import numpy as np
from Network_data import *
from Aux_fcns import *
import os
##############################


"""    Plot Confestion on Edges   """

SiouxNetwork, SiouxNetwork_data = Create_Network()

import pickle
file = open('stored_congestions.pckl', 'rb')
loader = pickle.load(file)

mean_congestions1 = loader[0]
mean_congestions2 = loader[1]
mean_congestions3 = loader[2]

cmap = plt.cm.Reds
print('Average Congestion shorthest path:  ' + str(np.mean(mean_congestions1)))
fig = Plot_Network(SiouxNetwork, SiouxNetwork_data, np.minimum(np.maximum(mean_congestions1/1, 0), 1000), 0 , 1000, cmap)
fig.tight_layout()
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
fig.show()

print('Average Congestion no regret:  ' + str(np.mean(mean_congestions2)))
fig = Plot_Network(SiouxNetwork, SiouxNetwork_data, np.minimum(np.maximum(mean_congestions2/1, 0),1000), 0 , 1000, cmap)
fig.tight_layout()
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
fig.show()


print('Average Congestion no routing:  ' + str(np.mean(mean_congestions3)))
fig = Plot_Network(SiouxNetwork, SiouxNetwork_data, np.minimum(np.maximum(mean_congestions3/1, 0),1000), 0 ,1000, cmap)
fig.tight_layout()
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
fig.show()


"""    Plot Regrets   """

file = open('stored_Regrets.pckl', 'rb')
loader = pickle.load(file)
file.close()

T = 100

All_avg_Regrets = loader[0]
All_std_Regrets = loader[1]
fig = plt.figure(figsize=(7.5,5.2))
plt.rcParams.update({'font.size': 14})
colors = ['green', 'gray', 'orange', 'black', 'deepskyblue', 'darkred', 'pink']
plt.plot(All_avg_Regrets[0], '--', color=colors[0])
markers = ['.', 's', 'x', '*', '^', 'o', 'd']
range_plot = [1,2,3,5]
for i in range_plot:
    p = plt.plot(All_avg_Regrets[i], marker = markers[i-1] , markersize = 10, markevery = 10, color = colors[i])
    color = p[0].get_color()
    plt.fill_between(range(T), All_avg_Regrets[i]+ All_std_Regrets[i], All_avg_Regrets[i] - All_std_Regrets[i], alpha = 0.2, color = color)

plt.title('Time averaged Regret')
plt.xlabel('time')
plt.ylim([10, 170])
plt.xlim([0,T-1])
plt.legend(['Hedge', 'Exp3', 'StackelUCB (poly dgr. 3)', 'StackelUCB (poly dgr. 4)', 'Always shorthest route'], prop={'size': 15})
fig.tight_layout()
fig.show()




