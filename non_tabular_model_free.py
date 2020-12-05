################ Tabular model-free algorithms ################
import numpy as np
import random


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        self.absorbing_state = self.env.absorbing_state

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)  # use seed to get same result every time
    eta = np.linspace(eta, 0, max_episodes)  # learning rate decaying appropriately
    epsilon = np.linspace(epsilon, 0, max_episodes)  # espilon decaying appropriately

    theta = np.zeros(env.n_features)  # initialisation of weights of the features
    timestep = 0
    for i in range(max_episodes):
        features = env.reset()  # start from a fresh environment
        q = features.dot(theta)  # get estimated value of state based on the features and current weights

        # Select action a for state s according to an e-greedy policy based on Q
        if timestep < env.n_actions:  # for the first 4 timesteps, choose each action once
            a = timestep  # select each action 0, 1, 2, 3 once
        else:
            # after having our first estimations, find the best action and break ties randomly
            best_action = randomBestAction(random_state, q)

            # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
            # (exploitation) or a random action (exploration)
            if random_state.random(1) < epsilon[i]:
                a = best_action  # use best action
            else:
                a = random_state.choice(range(env.n_actions))  # use random action
        timestep += 1

        done = False
        while not done:  # while not in absorbing state
            features_prime, r, done = env.step(a)  # Get next state and reward for the chosen action
            delta = r - q[a]  # compute the the difference between the observed reward and the estimated reward

            q = features_prime.dot(theta)  # get new estimated rewards

            # Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if timestep < env.n_actions:  # for the first 4 timesteps, choose each action once
                a_prime = timestep  # select each action 0, 1, 2, 3 once
            else:
                # after having our first estimations, find the best action and break ties randomly
                best_action = randomBestAction(random_state, q)

                # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
                # (exploitation) or a random action (exploration)
                if random_state.random(1) < epsilon[i]:
                    a_prime = best_action  # use best action
                else:
                    a_prime = random_state.choice(range(env.n_actions))  # use random action
            timestep += 1

            # Temporal difference
            delta += (gamma * q[a_prime])  # apply discount factor using the estimated value for the e-greedy policy
            theta += eta[i] * delta * features[a]  # update the weights based on gradient descent
            features = features_prime
            a = a_prime

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)  # use seed to get same result every time
    eta = np.linspace(eta, 0, max_episodes)  # learning rate decaying appropriately
    epsilon = np.linspace(epsilon, 0, max_episodes)  # espilon decaying appropriately

    theta = np.zeros(env.n_features)  # initialisation of weights of the features
    timestep = 0
    for i in range(max_episodes):
        features = env.reset()  # start from a fresh environment
        q = features.dot(theta)  # get estimated value of state based on the features and current weights

        done = False
        while not done:  # while not in absorbing state

            # Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if timestep < env.n_actions:  # for the first 4 timesteps, choose each action once
                a = timestep  # select each action 0, 1, 2, 3 once
            else:
                # after having our first estimations, find the best action and break ties randomly
                best_action = randomBestAction(random_state, q)

                # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
                # (exploitation) or a random action (exploration)
                if random_state.random(1) < epsilon[i]:
                    a = best_action  # use best action
                else:
                    a = random_state.choice(range(env.n_actions))  # use random action
            timestep += 1

            features_prime, r, done = env.step(a)  # Get next state and reward for the chosen action
            delta = r - q[a]  # compute the the difference between the observed reward and the estimated reward

            q = features_prime.dot(theta)  # get new estimated rewards
            # Temporal difference
            delta += (gamma * max(q))  # apply discount factor
            theta += eta[i] * delta * features[a]  # update the weights based on gradient descent
            features = features_prime

    return theta


def randomBestAction(random_state, mean_rewards):
    # get the best actions from mean_rewards
    best_actions = np.array(np.argwhere(mean_rewards == np.amax(mean_rewards))).flatten()
    return random_state.choice(best_actions, 1)[0]  # break ties randomly and return one of the best actions
