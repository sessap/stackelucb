import numpy as np
from Network_data import *
from Aux_fcns import *
import os
import GPy
##############################
np.random.seed(3)

SiouxNetwork, SiouxNetwork_data = Create_Network()
OD_demands = pandas.read_csv("SiouxFallsNet/SiouxFalls_OD_matrix.txt", header=None)
OD_demands = OD_demands.values
Strategy_vectors = Compute_Strategy_vectors(OD_demands, SiouxNetwork_data.Freeflowtimes, SiouxNetwork, SiouxNetwork_data.Edges)

Demands = []
for i in range(24):
    for j in range(24):
        if OD_demands[i, j] > 0:
            Demands.append(OD_demands[i, j] / 100)
Demands = np.array(Demands).reshape((-1,1))

Leader_demands = np.zeros((24,24))
Leader_demands[11,17] = 300*100

routes = 3
Strategies_Leader = Compute_Strategy_vectors(Leader_demands, SiouxNetwork_data.Freeflowtimes, SiouxNetwork, SiouxNetwork_data.Edges, routes, 10000)
Strategies_Leader = Strategies_Leader[0]
E = len(SiouxNetwork_data.Edges)
All_Strategies = []
vehicles_routed = []
for scale in [1, 0.75, 0.5, 0.25]:
    for i1 in range(routes):
        for i2 in range(routes):
            for i3 in range(routes):
                strategy =  1/3 * Strategies_Leader[i1]+  1/3 * Strategies_Leader[i2]+  1/3 * Strategies_Leader[i3]
                All_Strategies.append(strategy * scale)
                vehicles_routed.append(scale*np.max(Strategies_Leader[0]))

X_shorthest = All_Strategies[0]
All_Strategies.remove(All_Strategies[0])# remove shorthest route (add later as first strategy)
temp , idx_unique =   np.unique(np.array(All_Strategies).squeeze(), axis = 0, return_index=True, )
Strategies_Leader = [All_Strategies[i] for i in idx_unique]
vehicles_routed = [vehicles_routed[i] for i in idx_unique]

Strategies_Leader = [X_shorthest] + Strategies_Leader
vehicles_routed = [np.max(X_shorthest)] + vehicles_routed
Strategies_Leader.append(np.zeros((E,1))) # Add the null strategy (do not route any vehicle)
vehicles_routed.append(0)

def reward_leader(routed_vehicles, y):
    reward = routed_vehicles - 10*np.mean(y)
    return reward


K = len(Strategies_Leader)
T = 100
Runs = 10

min_payoff = 1000
max_payoff = 0
X = []
Y = []
for i in range(10000):
    scalings = np.random.rand(len(Strategy_vectors))
    a = np.random.randint(0,K)
    x = Strategies_Leader[a]
    new_times = np.multiply(SiouxNetwork_data.Freeflowtimes, np.power(np.ones((E,1)) + 0.15 * np.divide(x, SiouxNetwork_data.Capacities), 4))
    theta = routing_choice_of_others(Strategy_vectors, new_times, scalings)
    y = f_cong(x, theta, SiouxNetwork_data.Capacities)
    reward = reward_leader(vehicles_routed[a], y)
    if reward > max_payoff:
        max_payoff = reward
    if reward < min_payoff:
        min_payoff = reward
    X.append(np.vstack((x, np.multiply(scalings.reshape((-1,1)), Demands)))) #np.vstack((x,theta))) x + theta
    Y.append(np.mean(y))

# Normalization? 
y_min = 0#np.min(Y)
y_max = 1#np.max(Y)
X_min = 0*np.min(X, axis = 0).T
X_max = X_min + 1
#The above values are now set so that normalization does not happen

def normalize( X, X_min, X_max):
    return np.divide(X- X_min, (X_max - X_min))

sigma_e = 5
X_data = np.array(X[0:100]).squeeze()
Y_data = (np.array(Y[0:100]).squeeze()).reshape(-1,np.size(Y[0])) + sigma_e * np.random.randn(100, np.size(Y[0]))


All_Regrets_runs = []
All_CumulRewards_runs = []
for run in range(Runs):
    print('Run: ' + str(run) )
    All_Regrets = []
    All_CumulRewards= []
    use_GP = 0
    Algo_list = ['Hedge', 'EXP3P', 'StackelUCB_poly3', 'StackelUCB_poly4', 'StackelUCB_poly5' ]
    for Algorithm in Algo_list:
        np.random.seed(run)
        if Algorithm == 'Hedge':
            Algo = Hedge(K, T, min_payoff, max_payoff)
        elif Algorithm == 'EXP3P':
            Algo = EXP3P(K, T, min_payoff, max_payoff)
        elif 1:
            use_GP = 1
            Kernel = GPy.kern.Poly(input_dim=E + len(Demands), order= int(Algorithm[-1]))
            GP_init = GPy.models.GPRegression(normalize(X_data, X_min, X_max), normalize(Y_data.reshape((len(Y_data), 1)), y_min, y_max), Kernel, normalizer=False, noise_var= (sigma_e/(y_max - y_min))** 2)
            GP_init.Gaussian_noise.fix((sigma_e / (y_max - y_min)) ** 2)
            GP_init.optimize('bfgs', max_iters=200, messages=True)

            X_history = normalize(X_data[0:1, :], X_min, X_max)
            Y_history = normalize(Y_data[0:1].reshape((-1, np.size(Y[0]))), y_min, y_max)
            GP_model = GPy.models.GPRegression(X_history, Y_history, GP_init.kern, normalizer=False)
            GP_model.Gaussian_noise.fix((sigma_e / (y_max - y_min)) ** 2)
            Algo = Hedge(K, T, min_payoff, max_payoff)

        Rewards = np.zeros(T)
        Regrets = np.zeros(T)
        Played_actions = np.empty(T)
        Cum_rewards = np.zeros((T,K))
        Others = np.zeros((T,E))  # Network occupancy due to uncontrolled vehicles
        for t in range(T):
            scalings = np.random.rand(len(Strategy_vectors))
            a_t = Algo.sample_action()
            Played_actions[t] = a_t
            x_t = Strategies_Leader[a_t]

            new_times = np.multiply(SiouxNetwork_data.Freeflowtimes , np.power( np.ones((E,1)) + 0.15*np.divide(x_t,SiouxNetwork_data.Capacities )  , 4) )
            others_t = routing_choice_of_others(Strategy_vectors, new_times, scalings)
            Others[t,:] = others_t.reshape((1,E))

            y_t = f_cong(x_t, others_t, SiouxNetwork_data.Capacities)
            Rewards[t] = reward_leader(vehicles_routed[a_t], y_t)

            payoffs_hindsight = np.zeros(K)
            for a in range(K):
                x_a = Strategies_Leader[a]
                new_times = np.multiply(SiouxNetwork_data.Freeflowtimes, np.power(np.ones((E,1)) + 0.15 * np.divide(x_a, SiouxNetwork_data.Capacities), 4))
                theta = routing_choice_of_others(Strategy_vectors, new_times, scalings)
                y_a = f_cong(x_a, theta, SiouxNetwork_data.Capacities)
                payoffs_hindsight[a] = reward_leader(vehicles_routed[a], y_a)
            if t == 0:
                Cum_rewards[t,:] =  payoffs_hindsight
            else:
                Cum_rewards[t,:] = Cum_rewards[t-1,:] + payoffs_hindsight
            Regrets[t] = 1/(t+1) * ( np.max(Cum_rewards[t,:] ) - np.sum(Rewards) )

            """ Update Strategy """
            if Algorithm == 'Hedge':
                Algo.Update(payoffs_hindsight)
            elif Algorithm == 'EXP3P':
                Algo.Update(a_t, Rewards[t])
            elif use_GP:
                payoffs_ucb = np.zeros(K)
                payoffs_lcb = np.zeros(K)
                for a in range(K):
                    x_a = Strategies_Leader[a]
                    X = np.vstack( (x_a,  np.multiply(scalings.reshape((-1,1)), Demands)) )
                    mean_y_a , var_y_a = GP_model.predict(normalize(X.reshape((1,-1)), X_min, X_max))
                    mean_y_a = mean_y_a * (y_max - y_min) + y_min
                    var_y_a = var_y_a * ((y_max - y_min) ** 2)

                    lcb_y_a = mean_y_a - 0.5*np.sqrt(var_y_a)
                    ucb_y_a = mean_y_a + 0.5*np.sqrt(var_y_a)
                    payoffs_ucb[a] = reward_leader(vehicles_routed[a], lcb_y_a)
                    payoffs_lcb[a] = reward_leader(vehicles_routed[a], ucb_y_a)

                Algo.Update(payoffs_ucb)

                X_t = np.vstack((x_t, np.multiply(scalings.reshape((-1,1)), Demands)) )
                y_t = np.mean(f_cong(x_t, others_t, SiouxNetwork_data.Capacities))   + sigma_e*np.random.randn()

                X_history = np.vstack((X_history, normalize(X_t.T, X_min, X_max)))
                Y_history = np.vstack((Y_history, normalize(y_t, y_min, y_max)))
                GP_model.set_XY(X_history, Y_history)


        if 0:
            cmap = plt.cm.Reds
            congestions = [0.15 * np.power(np.divide( (Others[t,:].reshape((E,1)) + Strategies_Leader[0]).squeeze(), SiouxNetwork_data.Capacities.squeeze()), 4) for t in range(T)]
            mean_congestions1 = np.mean(congestions, axis = 0)
            print('Average Congestion shorthest path:  ' + str(np.mean(mean_congestions1)))
            fig = Plot_Network(SiouxNetwork, SiouxNetwork_data, np.maximum(np.log(mean_congestions1), np.log(0.25)), 0 , np.log(1000), cmap)
            fig.show()
            fig.tight_layout()

            congestions = [0.15 * np.power(np.divide( (Others[t,:].reshape((E,1)) + Strategies_Leader[int(Played_actions[t])]).squeeze(), SiouxNetwork_data.Capacities.squeeze()), 4) for t in range(T)]
            mean_congestions2 = np.mean(congestions, axis = 0)
            print('Average Congestion no regret:  ' + str(np.mean(mean_congestions2)))
            fig = Plot_Network(SiouxNetwork, SiouxNetwork_data, np.maximum(np.log(mean_congestions2), np.log(0.25)), 0 , np.log(1000), cmap)
            fig.show()
            fig.tight_layout()

            congestions = [0.15 * np.power(np.divide( (Others[t,:].reshape((E,1)) + Strategies_Leader[-1]).squeeze(), SiouxNetwork_data.Capacities.squeeze()), 4) for t in range(T)]
            mean_congestions3 = np.mean(congestions, axis = 0)
            print('Average Congestion no routing:  ' + str(np.mean(mean_congestions3)))
            fig = Plot_Network(SiouxNetwork, SiouxNetwork_data, np.maximum(np.log(mean_congestions3), np.log(0.25)), 0 , np.log(1000), cmap)
            fig.show()
            fig.tight_layout()

            import pickle
            file = open('stored_congestions.pckl', 'wb')
            pickle.dump((mean_congestions1, mean_congestions2, mean_congestions3), file)
            file.close()

        All_CumulRewards.append(np.sum(Rewards))
        All_Regrets.append(Regrets)
        del(Algo)
        if use_GP:
            del(GP_model)
            del(GP_init)

    # Regret and Cumulative rewards of Strategy: always shorthest route
    All_Regrets.append( np.divide(np.max(Cum_rewards, axis = 1) - Cum_rewards[:,0], np.arange(1,T+1) )  )
    All_CumulRewards.append(Cum_rewards[-1,0])
    # Regret and Cumulative rewards of Strategy: no routing
    All_Regrets.append( np.divide(np.max(Cum_rewards, axis = 1) - Cum_rewards[:,-1], np.arange(1,T+1) )  )
    All_CumulRewards.append(Cum_rewards[-1,-1])

    All_Regrets_runs.append(All_Regrets)
    All_CumulRewards_runs.append(All_CumulRewards)


All_avg_Regrets = np.mean(All_Regrets_runs, axis = 0)
All_std_Regrets = np.std(All_Regrets_runs, axis = 0)
All_avg_cumRewards = np.mean(All_CumulRewards_runs, axis = 0)


if 0:
    import pickle
    file = open('stored_Regrets.pckl', 'wb')
    pickle.dump((All_avg_Regrets, All_std_Regrets, All_avg_cumRewards), file)
    file.close()

    
print('Cumulative Reward our approach: ' + str(All_avg_cumRewards[2]))
print('Cumulative Reward no routing: ' + str(All_avg_cumRewards[-1]))
print('Cumulative Reward always shorthest route: ' + str(All_avg_cumRewards[-2]))