################ Tabular model-free algorithms ################
import numpy as np
import random

def randomBestAction(random_state, mean_rewards):
    # get the best actions from mean_rewards
    best_actions = np.array(np.argwhere(mean_rewards == np.amax(mean_rewards))).flatten()
    return random_state.choice(best_actions, 1)[0]  # break ties randomly and return one of the best actions

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes) #learning_rate?
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    timestep = 0
    for i in range(max_episodes):
        s = env.reset()
        ###### start of our code #######
        #Select action a for state s according to an e-greedy policy based on Q
        if(timestep < env.n_actions):
            a = timestep #select each action once
        else:
            best_action = randomBestAction(random_state, np.average(q, axis=0))
            if(random_state.random(1) < epsilon[i]):
                a = random_state.choice(range(env.n_actions))
            else:
                a = best_action
        timestep += 1

        while(s != env.absorbing_state):
            s_prime, r, done = env.step(a)

            #Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if(timestep < env.n_actions):
                a_prime = timestep #select each action once
            else:
                best_action = randomBestAction(random_state, np.average(q, axis=0))
                if(random_state.random(1) < epsilon[i]):
                    a_prime = random_state.choice(range(env.n_actions))
                else:
                    a_prime = best_action
            timestep += 1
            q[s,a] = q[s,a] + eta[i] * (r + gamma * q[s_prime, a_prime] - q[s,a])
            s = s_prime
            a = a_prime
        ###### end of our code ##########
    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes) #learning_rate?
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    sum_rewards = np.zeros(env.n_actions)
    times_action_observed = np.zeros(env.n_actions)
    timestep = 0
    for i in range(max_episodes):
        s = env.reset()
        while(s != env.absorbing_state):
            #Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if(timestep < env.n_actions):
                a = timestep #select each action once
            else:
                best_action = randomBestAction(random_state, np.average(q, axis=0))
                if(random_state.random(1) < epsilon[i]):
                    a = random_state.choice(range(env.n_actions))
                else:
                    a = best_action
            timestep += 1

            s_prime, r, done = env.step(a)
            a_prime = randomBestAction(random_state, np.average(q, axis=0))

            q[s,a] += eta[i] * (r + gamma * q[s_prime, a_prime] - q[s,a])
            s = s_prime
            a = a_prime
        ###### end of our code ##########
    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value