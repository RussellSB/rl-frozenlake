################ Tabular model-free algorithms ################
import numpy as np
import random

def randomBestAction(random_state, mean_rewards):
    # get the best actions from mean_rewards
    best_actions = np.array(np.argwhere(mean_rewards == np.amax(mean_rewards))).flatten()
    return random_state.choice(best_actions, 1)[0]  # break ties randomly and return one of the best actions

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)  # use seed to get same result every time
    eta = np.linspace(eta, 0, max_episodes)  # learning rate decaying appropriately
    epsilon = np.linspace(epsilon, 0, max_episodes)  # espilon decaying appropriately

    q = np.zeros((env.n_states, env.n_actions)) # estimated value for each state and action
    timestep = 0
    for i in range(max_episodes):
        s = env.reset()  # start from a fresh environment

        #Select action a for state s according to an e-greedy policy based on Q
        if(timestep < env.n_actions):  # for the first 4 timesteps, choose each action once
            a = timestep  # select each action 0, 1, 2, 3 once
        else:
            # after having our first estimations, find the best action and break ties randomly
            best_action = randomBestAction(random_state, q[s])

            # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
            # (exploitation) or a random action (exploration)
            if(random_state.random(1) < epsilon[i]):
                a = random_state.choice(range(env.n_actions))  # use random action
            else:
                a = best_action  # use best action
        timestep += 1

        while(s != env.absorbing_state):  # while not in absorbing state
            s_prime, r, done = env.step(a)

            #Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if(timestep < env.n_actions):  # for the first 4 timesteps, choose each action once
                a_prime = timestep  # select each action 0, 1, 2, 3 once
            else:
                # after having our first estimations, find the best action and break ties randomly
                best_action = randomBestAction(random_state, q[s_prime])

                # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
                # (exploitation) or a random action (exploration)
                if(random_state.random(1) < epsilon[i]):
                    a_prime = random_state.choice(range(env.n_actions))  # use random action
                else:
                    a_prime = best_action  # use best action
            timestep += 1

            # update estimated value of the current state and action
            q[s,a] += eta[i] * (r + gamma * q[s_prime, a_prime] - q[s,a])
            s = s_prime
            a = a_prime

    policy = q.argmax(axis=1)  # get policy from the best values of q
    value = q.max(axis=1)

    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)  # use seed to get same result every time
    eta = np.linspace(eta, 0, max_episodes)  # learning rate decaying appropriately
    epsilon = np.linspace(epsilon, 0, max_episodes)  # espilon decaying appropriately

    q = np.zeros((env.n_states, env.n_actions)) # estimated value for each state and action
    timestep = 0
    for i in range(max_episodes):
        s = env.reset()  # start from a fresh environment

        while(s != env.absorbing_state):  # while not in absorbing state

            #Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if(timestep < env.n_actions):  # for the first 4 timesteps, choose each action once
                a = timestep  # select each action 0, 1, 2, 3 once
            else:
                # after having our first estimations, find the best action and break ties randomly
                best_action = randomBestAction(random_state, q[s])

                # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
                # (exploitation) or a random action (exploration)
                if(random_state.random(1) < epsilon[i]):
                    a = random_state.choice(range(env.n_actions))  # use random action
                else:
                    a = best_action  # use best action
            timestep += 1

            s_prime, r, done = env.step(a)  # Get next state and reward for the chosen action

            q_max = max(q[s_prime])  # find the best action for next step
            # update estimated value of the current state and action
            q[s,a] += eta[i] * (r + gamma * q_max - q[s,a])
            s = s_prime
    policy = q.argmax(axis=1)  # get policy from the best values of q
    value = q.max(axis=1)

    return policy, value