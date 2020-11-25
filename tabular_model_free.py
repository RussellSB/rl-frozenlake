################ Tabular model-free algorithms ################
import numpy as np
import random

def randomBestAction(mean_rewards):
    best_actions = np.array(np.argwhere(mean_rewards == np.amax(mean_rewards))).flatten()
    return np.random.choice(best_actions, 1)[0]

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes) #learning_rate?
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    sum_rewards = np.zeros(env.n_actions)
    times_action_observed = np.zeros(env.n_actions)
    timestep = 0
    for i in range(max_episodes):
        s = env.reset()
        ###### start of our code #######
        #Select action a for state s according to an e-greedy policy based on Q
        if(timestep < env.n_actions):
            a = timestep #select each action once
            #print(f'sum_rewards = {sum_rewards}')
        else:
            mean_rewards = sum_rewards / times_action_observed
            best_action = randomBestAction(mean_rewards)
            if(np.random.random(1) < epsilon[i]):
                a = best_action
            else:
                a = random.randrange(env.n_actions)
            #print(f'mean_rewards = {mean_rewards}')
            #print(f'best_action = {best_action}')
        timestep += 1

        while(s != env.absorbing_state):
            #print(f'action = {a}')
            s_prime, r, done = env.step(a)
            sum_rewards[a] += r
            times_action_observed[a] += 1

            #Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if(timestep < env.n_actions):
                a_prime = timestep #select each action once
                #print(f'sum_rewards = {sum_rewards}')
            else:
                mean_rewards = sum_rewards / times_action_observed
                best_action = randomBestAction(mean_rewards)
                if(np.random.random(1) < epsilon[i]):
                    a_prime = best_action
                else:
                    a_prime = random.randrange(env.n_actions)
                #print(f'mean_rewards = {mean_rewards}')
                #print(f'best_action = {best_action}')
            timestep += 1
            q[s,a] = q[s,a] + eta[i] * (r + gamma * max * q[s_prime, a_prime] - q[s,a])
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