import numpy as np
import matplotlib.pyplot as plt
import GPy
from scipy.spatial import distance
import scipy.stats

np.random.seed(3)


E = 5
Cells_centers = []
for x1 in np.arange(0.5,E+0.5,1):
    for x2 in np.arange(0.5, E + 0.5, 1):
        Cells_centers.append([x1,x2])
C = len(Cells_centers)

X = []
for i in range(500):
    u = np.zeros(C+1)
    u[1:C] = np.random.rand(C-1)
    u[C] = 1
    u.sort()
    x = u[1:C+1]- u[0:C]
    X.append(x)
for c in range(C):
    x = np.zeros(C)
    x[c] = 1
    X.append(x)
K = len(X)

# For computational tractability the park 2D coordinates are discretized with 0.1 precision
Y = []
for y1 in np.arange(0, E, 0.1):
    for y2 in np.arange(0, E, 0.1):
        Y.append([y1,y2])
Y_len = len(Y)


def findcell(y):
    for c in range(C):
        if Cells_centers[c][0]-0.5 <= y[0] and y[0] < Cells_centers[c][0]+0.5 and Cells_centers[c][1]-0.5 <= y[1] and y[1]< Cells_centers[c][1]+0.5:
            cell_idx = c
            break
    return int(cell_idx)


def animal_density_fun(location):
    mean1 = np.array([1.5, 3.5])
    cov1 = 4*np.array([[0.2, 0.1], [0.1, 0.2]])
    mean2 = np.array([3.5, 1.5])
    cov2 = 4*np.array([[0.2, 0.1], [0.1, 0.2]])
    mean3 = np.array([4, 4])
    cov3 = 4*np.array([[0.2, 0.1], [0.1, 0.2]])
    mean4 = np.array([1.5, 1.5])
    cov4 = 4*np.array([[0.6, 0.5], [0.5, 0.6]])
    density = 0.25*scipy.stats.multivariate_normal(mean1, cov1).pdf(location) + 0.25*scipy.stats.multivariate_normal(mean2, cov2).pdf(location) + 0.25*scipy.stats.multivariate_normal(mean3, cov3).pdf(location) + 0.25*scipy.stats.multivariate_normal(mean4, cov4).pdf(location)
    return density


Phi = np.empty(Y_len)
for y_idx in range(Y_len):
    Phi[y_idx] = animal_density_fun(Y[y_idx])
Phi = 10*Phi

Phi_lowerdim = np.zeros(C)  # Feature vector theta used by the rangers to learn about the poachers' responses
for y_idx in range(Y_len):
    c = findcell(Y[y_idx])
    Phi_lowerdim[c] = np.maximum(Phi_lowerdim[c], animal_density_fun(Y[y_idx]) )


plt.imshow(Phi.reshape((E*10,E*10)).T, cmap='binary', interpolation='nearest',origin='lower')
plt.show()

poacher_init_loc = np.array([0,0])
Distances = np.array([ distance.euclidean(poacher_init_loc, Y[i]) for i in range(Y_len)])
R_a =  Phi  - 0.5*Distances* 1/(np.max(Distances))
P_a = -1 # Doesn't depend on where the poachers attack, but only if they get caught

w1 = -3
w2 = 1
w3 = 1
delta = 2
gamma = 3
def SU(x, y):
    c = findcell(y)
    y_idx = Y.index(y)
    f_x = delta*x[c]**gamma/(delta*x[c]**gamma + (1-x[c])**gamma)
    SU =  w1* f_x + w2*R_a[y_idx] + w3* P_a
    return SU

Best_responses_idx = np.empty(K, 'i')
for k in range(K):
    Best_responses_idx[k] = np.argmax([SU(X[k], Y[y_idx]) for y_idx in range(Y_len)])

R_d = 1
def r(x,y):
    c = findcell(y)
    animal_density = animal_density_fun(y)
    reward = R_d*x[c] + (1-x[c])*(- animal_density)
    return reward

Rewards_vectorized = np.array([r(X[k], Y[int(Best_responses_idx[k])]) for k in range(K)])
OPT_discrete = np.max(Rewards_vectorized)
x_OPT_idx = np.argmax(Rewards_vectorized)
x_OPT = X[x_OPT_idx]

if 1:
    fig = plt.figure(figsize=(4,4))
    plt.imshow(Phi.reshape((E*10,E*10)).T, cmap='binary', interpolation='nearest', origin='lower', extent=[0, 50, 0, 50], alpha = 1)
    plt.contour(Phi.reshape((E*10,E*10)).T, colors = 'black', linewidth = 0.1, alpha = 0.2)
    plt.scatter(poacher_init_loc[0],poacher_init_loc[1],color='r', marker = 's', s = 200)
    frame1 = plt.gca()
    frame1.set_xticks(np.arange(0,50,10))
    frame1.set_yticks(np.arange(0,50,10))
    plt.grid(True,color='black', linestyle = ':', linewidth = 0.5)
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.xlim([0,50])
    plt.ylim([0,50])
    repelem = lambda a, x, y: np.repeat(np.repeat(a, x, axis=0), y, axis=1)
    plt.imshow(repelem(x_OPT.reshape((E,E)).T, 10,10), cmap='Greens', interpolation='nearest', origin='lower', extent=[0, 50, 0, 50], alpha = 0.3)
    fig.show()

y_min = np.array([0,0])
y_max = np.array([5, 5])

sigma_e = 0.1
Kernel =   GPy.kern.Matern52(input_dim=2*C, lengthscale=0.2, variance =3)
N_init_data = 100 # for offline hyperparameters optimization


dim_x = C+C
dim_y = 2
X_data_Algo1 = np.empty((N_init_data, dim_x))
Y_data_Algo1_noiseless = np.empty((N_init_data, dim_y))
Y_data_Algo1 = np.empty((N_init_data, dim_y))
for i in range(N_init_data):
    k = np.random.randint(0, K)
    X_data_Algo1[i,:] = np.hstack((X[k] , Phi_lowerdim))
    y = Y[int(Best_responses_idx[k])] + sigma_e * np.random.randn(1, dim_y)
    Y_data_Algo1[i,:] = y.squeeze()
    Y_data_Algo1_noiseless[i,:] = Y[int(Best_responses_idx[k])]

GP_init = GPy.models.GPRegression(X_data_Algo1,Y_data_Algo1, Kernel, normalizer= False, noise_var = sigma_e**2)
GP_init.Gaussian_noise.fix(sigma_e**2)
GP_init.optimize('bfgs', max_iters=200,messages=True)


Y_responses = np.empty((K,2))
for k in range(K):
    Y_responses[k,:] = Y[Best_responses_idx[k]]


""" Find best strategy using offline data """
print('Finding best strategy offline ...')
np.random.seed(3)
N_offline_data = 1000
Reward_best_offline = np.empty((1,1))
for run in range(1):
    X_data_Algo1 = np.empty((N_offline_data, dim_x))
    Y_data_Algo1 = np.empty((N_offline_data, dim_y))
    for i in range(N_offline_data):
        k = np.random.randint(0,K)
        X_data_Algo1[i,:] = np.hstack((X[k] , Phi_lowerdim))
        y = Y[int(Best_responses_idx[k])] + sigma_e * np.random.randn(1, dim_y)
        Y_data_Algo1[i,:] = y.squeeze()

    GP_Algo1 = GPy.models.GPRegression(X_data_Algo1, Y_data_Algo1, GP_init.kern, normalizer=False, noise_var=sigma_e ** 2)
    Y_mean, Y_var = GP_Algo1.predict(np.hstack((np.vstack(X), np.tile(Phi_lowerdim, (K, 1)))))  # GP_Algo1.predict(np.vstack(X))
    Y_mean = np.minimum(np.array(Y_mean), y_max-1e-3)
    Y_mean = np.maximum(Y_mean, y_min+1e-3)
    idx_x_max = int(np.argmax([r(X[k], Y_mean[k, :]) for k in range(K)]))
    reward_best = Rewards_vectorized[idx_x_max]

    Reward_best_offline[run] = reward_best

if 1:
    for beta_t in [0.01, 0.05, 0.1]:
        """  Find best strategy online   """
        print('Finding best strategy online using the proposed Algorithm...')
        T = 100
        Runs = 10
        # Start game
        Rewards_runs = []
        for run  in range(Runs):
            X_data_Algo1 = X_data_Algo1[0:1, :]
            Y_data_Algo1 = Y_data_Algo1[0:1, :]
            GP_Algo1.set_XY(X_data_Algo1,Y_data_Algo1)

            Rewards = np.empty(T)
            for t in range(T):
                payoffs_ucb = np.empty(K)
                payoffs_lcb = np.empty(K)
                Y_mean , Y_var = GP_Algo1.predict( np.hstack(( np.vstack(X), np.tile(Phi_lowerdim, (K,1))) ))

                for k in range(K):
                    y_mean = Y_mean[k,:]
                    y_var = Y_var[k,:]
                    for eta1 in [-beta_t, 0, beta_t]:
                        for eta2 in [-beta_t, 0, beta_t]:
                            max_payoff = -1000
                            min_payoff = 1000
                            y_cb = y_mean + np.multiply([eta1, eta2], np.sqrt(y_var))
                            y_cb = np.minimum(y_cb, y_max-1e-3)
                            y_cb = np.maximum(y_cb, y_min+1e-3)
                            payoff = r(X[k], y_cb.squeeze())
                            if payoff > max_payoff:
                                max_payoff = payoff
                            if payoff < min_payoff:
                                min_payoff = payoff

                    payoffs_ucb[k] = max_payoff
                    payoffs_lcb[k] = min_payoff

                idx_x_t = np.argmax(payoffs_ucb)
                x_t = X[idx_x_t]
                y_t = Y[Best_responses_idx[idx_x_t]]
                Rewards[t] = Rewards_vectorized[idx_x_t]

                X_data_Algo1 = np.vstack((X_data_Algo1, np.hstack((x_t , Phi_lowerdim))))
                Y_data_Algo1 = np.vstack((Y_data_Algo1, y_t + sigma_e* np.random.randn(1,dim_y)))
                GP_Algo1.set_XY(X_data_Algo1, Y_data_Algo1)
                print(t)

            Rewards_runs.append(Rewards)
            print('run = '+ str(run))

        import pickle

        file = open('stored_rewards_OurAlgo_beta_' + str(beta_t) + '.pckl', 'wb')
        pickle.dump(Rewards_runs, file)
        file.close()


""" Find strategy online with GP-UCB (modeling the whole function as a black-box)"""
sigma = 0.002
Kernel =   GPy.kern.Matern52(input_dim=2*C, lengthscale=0.2, variance =3)

dim_x = C+C
dim_y = 1
X_data_Algo2 = np.empty((N_init_data, dim_x))
Y_data_Algo2 = np.empty((N_init_data, dim_y))
for i in range(N_init_data):
    k = np.random.randint(0,len(X))
    X_data_Algo2[i,:] = np.hstack((X[k] , Phi_lowerdim))
    y = Rewards_vectorized[k] + sigma * np.random.randn(1, dim_y)
    Y_data_Algo2[i,:] = y.squeeze()

GP_init = GPy.models.GPRegression(X_data_Algo2,Y_data_Algo2, Kernel, normalizer= None, noise_var = sigma**2)
GP_init.Gaussian_noise.fix(sigma**2)
#GP_init.optimize_restarts(num_restarts=10)
GP_init.optimize('bfgs', max_iters=200,messages=True)

GP_Algo2 = GPy.models.GPRegression(X_data_Algo2, Y_data_Algo2, GP_init.kern, normalizer=None, noise_var=sigma ** 2)

plt.figure()
plt.plot(Y_data_Algo2)
Y_mean, Y_var = GP_init.predict(X_data_Algo2)
plt.plot(Y_mean)
lcb = Y_mean - 0.2 * np.sqrt(Y_var)
ucb = Y_mean + 0.2 * np.sqrt(Y_var)
plt.fill_between(range(N_init_data), lcb.squeeze(), ucb.squeeze(), color='gray', alpha=0.2)
plt.show()

for beta_t in [0.2, 0.5, 1, 3, 5, 7, 9]:
    print('Finding best strategy online using GP-UCB ...')
    T = 100
    Runs = 10
    # Start game
    Rewards_ucb_runs = []
    for run  in range(Runs):
        X_data_Algo2 = np.empty((N_init_data, dim_x))
        Y_data_Algo2 = np.empty((N_init_data, dim_y))
        for i in range(N_init_data):
            k = np.random.randint(0, len(X))
            X_data_Algo2[i, :] = np.hstack((X[k], Phi_lowerdim))
            y = Rewards_vectorized[k] + sigma * np.random.randn(1, dim_y)
            Y_data_Algo2[i, :] = y.squeeze()

        GP_init = GPy.models.GPRegression(X_data_Algo2, Y_data_Algo2, Kernel, normalizer=None, noise_var=sigma ** 2)
        GP_init.Gaussian_noise.fix(sigma ** 2)
        # GP_init.optimize_restarts(num_restarts=10)
        GP_init.optimize('bfgs', max_iters=200, messages=True)

        GP_Algo2 = GPy.models.GPRegression(X_data_Algo2, Y_data_Algo2, GP_init.kern, normalizer=None, noise_var=sigma ** 2)

        X_data_Algo2 = X_data_Algo2[0:1, :]
        Y_data_Algo2 = Y_data_Algo2[0:1, :]
        GP_Algo2.set_XY(X_data_Algo2,Y_data_Algo2)

        Rewards = np.empty(T)
        for t in range(T):
            payoffs_ucb = np.empty(K)
            payoffs_lcb = np.empty(K)
            Y_mean , Y_var = GP_Algo2.predict( np.hstack(( np.vstack(X), np.tile(Phi_lowerdim, (K,1))) ))

            if 0:
                plt.figure()
                plt.plot(Rewards_vectorized)
                plt.plot(Y_mean)
                lcb = Y_mean - 0.2*np.sqrt(Y_var)
                ucb = Y_mean + 0.2*np.sqrt(Y_var)
                plt.fill_between(range(len(X)),lcb.squeeze() , ucb.squeeze(), color = 'gray', alpha = 0.2)
                plt.show()

            for k in range(K):
                payoffs_ucb[k] = Y_mean[k] + beta_t*np.sqrt(Y_var[k])

            idx_x_t = np.argmax(payoffs_ucb)
            x_t = X[idx_x_t]
            Rewards[t] = Rewards_vectorized[idx_x_t]

            X_data_Algo2 = np.vstack((X_data_Algo2, np.hstack((x_t , Phi_lowerdim))))
            Y_data_Algo2 = np.vstack((Y_data_Algo2, Rewards_vectorized[idx_x_t] + sigma* np.random.randn(1,dim_y)))
            GP_Algo2.set_XY(X_data_Algo2, Y_data_Algo2)
            print(t)

        Rewards_ucb_runs.append(Rewards)
        print('run = '+ str(run))

    import pickle
    file = open('stored_rewards_GPUCB_beta_'+ str(beta_t)+'.pckl', 'wb')
    pickle.dump(Rewards_ucb_runs, file)
    file.close()


plt.figure()
plt.plot(range(T), OPT_discrete*np.ones(T))
#plt.plot(np.mean(Rewards_runs, axis = 0))
for beta_t in [0.2, 0.5, 1, 3, 5, 7, 9]:
    file = open('stored_rewards_GPUCB_beta_'+ str(beta_t)+'.pckl', 'rb')
    loader = pickle.load(file)
    file.close()
    plt.plot(np.mean(loader, axis = 0))
plt.show()


print('Finding max-min strategy ...')
Payoff_matrix = np.zeros((K, Y_len))
for k in range(K):
    for y in range(Y_len):
        Payoff_matrix[k,y] = r(X[k], Y[y])

Max_Min_OPT_discrete = np.max(np.min(Payoff_matrix, axis = 1))

plt.plot(range(T), Max_Min_OPT_discrete*np.ones(T))
plt.show()



if 0:
    import pickle
    file = open('stored_rewards.pckl', 'wb')
    pickle.dump((OPT_discrete ,Rewards_runs,Max_Min_OPT_discrete, Reward_best_offline, Rewards_ucb_runs), file)
    file.close()


if 0:
    file = open('stored_rewards.pckl', 'rb')
    loader = pickle.load(file)
    file.close()

    fig = plt.figure(figsize=(7,4.5))
    plt.rcParams.update({'font.size': 14})
    plt.plot(range(T), loader[0]*np.ones(T), linestyle = '--', color = 'green')
    plt.plot(range(T), loader[2]*np.ones(T), linestyle = '-', marker = 'o', markevery = 5, color = 'red')
    plt.plot(range(T), loader[3]*np.ones(T), linestyle = '-', marker = 'x', markevery = 5, color = 'deepskyblue')
    plt.plot(np.mean(loader[1], axis = 0) , linestyle = '-', marker = '*', markevery = 5, color = 'black')
    plt.legend(['OPT', 'Max-Min', 'Best offline', 'our Algorithm'],  prop={'size': 15})
    plt.fill_between(range(T),  np.mean(loader[1], axis = 0)-  np.std(loader[1], axis = 0)  , np.mean(loader[1], axis = 0) + np.std(loader[1], axis = 0) , alpha = 0.2, color = 'black')
    plt.title('Obtained Rewards')
    plt.xlabel('rounds')
    fig.tight_layout()
    fig.show()

