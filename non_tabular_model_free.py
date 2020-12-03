################ Tabular model-free algorithms ################
import numpy as np
import random

class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

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
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        q = features.dot(theta)

        # TODO:

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    s = env.reset()
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)
    timestep = 0
    for i in range(max_episodes):
        features = env.reset()

        # TODO:

        # Compute Q(a)
        q = features.dot(theta)
        print('Q:')
        print(q)
        print('Features:')
        print(features)

        while (s != env.absorbing_state):
            # Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if (timestep < env.n_actions):
                a = timestep  # select each action once
            else:
                best_action = randomBestAction(np.average(q, axis=0))
                if (np.random.random(1) < epsilon[i]):
                    a = best_action
                else:
                    a = random.randrange(env.n_actions)
            timestep += 1

            # Get next state and reward for best action chosen
            s_prime, r, done = env.step(a)
            delta = r - q[a]

            q = features.dot(theta)

            #Temporal difference
            delta = delta + (gamma * randomBestAction(np.average(q, axis=0)))
            theta = theta + (eta * delta.dot(features))

    return theta

def randomBestAction(mean_rewards):
    best_actions = np.array(np.argwhere(mean_rewards == np.amax(mean_rewards))).flatten()
    return np.random.choice(best_actions, 1)[0]