import numpy as np
import matplotlib.pyplot as plt
import pickle

T = 100

file = open('stored_rewards_OurAlgo_beta_'+ str(0.1)+'.pckl', 'rb')
loader = pickle.load(file)
Rewards_ouralgo = loader
file.close()

file = open('stored_rewards.pckl', 'rb')
loader = pickle.load(file)
file.close()

fig = plt.figure(figsize=(7,4.5))
plt.rcParams.update({'font.size': 14})
plt.plot(range(T), loader[0]*np.ones(T), linestyle = '--', color = 'green', linewidth = 2)
plt.plot(range(T), loader[2]*np.ones(T), linestyle = '-', marker = 'o', markevery = 5, color = 'red')
plt.plot(range(T), np.mean(loader[3], axis = 0)*np.ones(T), linestyle = '-', marker = 'x', markevery = 5, color = 'deepskyblue')
plt.plot(np.mean(Rewards_ouralgo, axis = 0) , linestyle = '-', marker = '*', markevery = 5, color = 'black')
plt.legend(['OPT', 'Max-Min', 'Best offline', 'our Algorithm'],  prop={'size': 15})
plt.fill_between(range(T),  np.mean(Rewards_ouralgo, axis = 0)-  np.std(Rewards_ouralgo, axis = 0)  , np.mean(Rewards_ouralgo, axis = 0) + np.std(Rewards_ouralgo, axis = 0) , alpha = 0.2, color = 'black')
plt.title('Obtained Rewards')
plt.xlabel('rounds')
fig.tight_layout()
fig.show()
fig.savefig('rewards.png')


OPT = loader[0]
Markers = ['*', 'd', 's', 'o', 'v']
fig = plt.figure(figsize=(7,4.5))
betas_ucb = [0.2, 0.5, 1, 3, 5]
betas_ouralgo = [0.01, 0.05, 0.1, 0.2, 0.5]
plt.plot(range(T), OPT*np.ones(T), linestyle = '--', color = 'green', linewidth = 2, label = 'OPT')
for beta_t in betas_ucb:
    file = open('stored_rewards_GPUCB_beta_'+ str(beta_t)+'.pckl', 'rb')
    loader = pickle.load(file)
    file.close()
    plt.plot(np.mean(loader, axis = 0), linestyle = '-', marker = Markers[betas_ucb.index(beta_t)], markevery = 5, label = r'$\beta_t =$'+ str(beta_t))
plt.legend(loc = 'upper left')
plt.title('Obtained rewards using GP-UCB')
plt.xlabel('rounds')
fig.tight_layout()
fig.show()
fig.savefig('rewards_GPUCB.png')


fig = plt.figure(figsize=(7,4.5))
plt.plot(range(T), OPT*np.ones(T), linestyle = '--', color = 'green', linewidth = 2, label = 'OPT')
for beta_t in betas_ouralgo:
    file = open('stored_rewards_OurAlgo_beta_'+ str(beta_t)+'.pckl', 'rb')
    loader = pickle.load(file)
    file.close()
    plt.plot(np.mean(loader, axis = 0), linestyle = '-', marker = Markers[betas_ouralgo.index(beta_t)], markevery = 5, label = r'$\beta_t =$'+ str(beta_t))
plt.legend(loc = 'upper left')
plt.title('Obtained rewards using the proposed algorithm')
plt.xlabel('rounds')
fig.tight_layout()
fig.show()
fig.savefig('rewards_ourAlgo.png')



#
