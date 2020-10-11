import numpy as np


def f_cong(x, theta, edge_capacities):
    total_occupancies = x+theta
    congestions = 0.15 * np.power(np.divide(total_occupancies, edge_capacities), 4)
    return congestions


def routing_choice_of_others(Strategy_vectors, traveltimes, scalings):
    E = len(Strategy_vectors[0][0]) #number of edges in the network
    N = len(Strategy_vectors) #number of OD pairs
    total_occupancies = np.zeros((E,1))
    for i in range(N):
        if 1:#np.random.binomial(1,0.5)
            K_i = np.minimum(len(Strategy_vectors[i]) , 2)
            a = np.argmin( [ np.dot(  Strategy_vectors[i][k].T, traveltimes)  for k in range(K_i)])
            scale = scalings[i]
            occupancies_i = scale*Strategy_vectors[i][a]
            total_occupancies = total_occupancies + occupancies_i
    return total_occupancies

class Hedge:
    def __init__(self, K, T, min_payoff, max_payoff):
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.weights = np.ones(K)
        self.T = T
        self.gamma_t = np.sqrt(8 * np.log(K) / T)

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))

    def Update(self, payoffs):
        # assert  all(payoffs > self.min_payoff*np.ones(self.K) - 1e-2) and  all(payoffs < self.max_payoff*np.ones(self.K) - 1e-2), "min payoff = "+ str(np.min(payoffs)) + " lb = " + str(self.min_payoff)
        payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
        payoffs = np.minimum(payoffs, self.max_payoff * np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.max_payoff - self.min_payoff))
        losses = np.ones(self.K) - np.array(payoffs_scaled)
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights / np.sum(self.weights)  # To avoid numerical errors when the weights become too small

class EXP3P:
    def __init__(self, K, T, min_payoff, max_payoff):
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.T = T
        self.weights = np.ones(K)
        self.rewards_est = np.zeros(K)

        self.beta = 0#np.sqrt(np.log(self.K) / (self.T * self.K))
        self.gamma = 0#1.05 * np.sqrt(np.log(self.K) * self.K / self.T)
        self.eta =  np.sqrt(np.log(self.K) / (self.T * self.K)) #*0.95
        #assert self.K == 1 or (self.beta > 0 and self.beta < 1 and self.gamma > 0 and self.gamma < 1)

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))

    def Update(self, played_a, payoff):
        prob = self.weights[played_a] / np.sum(self.weights)
        # assert  payoff > self.min_payoff - 1e-2 and payoff < self.max_payoff +1e-2, "min payoff = "+ str(payoff) + " lb = " + str(self.min_payoff)
        payoff = np.maximum(payoff, self.min_payoff)
        payoff = np.minimum(payoff, self.max_payoff)
        payoff_scaled = np.array((payoff - self.min_payoff) / (self.max_payoff - self.min_payoff))

        self.rewards_est = self.rewards_est + self.beta * np.divide(np.ones(self.K),
                                                                    self.weights / np.sum(self.weights))
        self.rewards_est[played_a] = self.rewards_est[played_a] + payoff_scaled / prob

        self.weights = np.exp(np.multiply(self.eta, self.rewards_est))
        self.weights = self.weights / np.sum(self.weights)
        self.weights = (1 - self.gamma) * self.weights + self.gamma / self.K * np.ones(self.K)