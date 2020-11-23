################ Tabular model-free algorithms ################
import numpy as np

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes) #learning_rate?
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        ###### start of our code #######
        #Select action a for state s according to an e-greedy policy based on Q
        a = 0 #TODO
        while(s != env.absorbing_state):
            s_prime, r, done = env.step(a)
            #Select action a_prime for state s_prime according to an e-greedy policy based on Q
            a_prime = 0 #TODO
            q[s,a] = q[s,a] + eta[i] * (r + gamma * q[s_prime, a_prime] - q[s,a])
            s = s_prime
            a = a_prime
        ###### end of our code ##########
    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        # TODO:

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value