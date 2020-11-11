import numpy as np
import contextlib
from itertools import product

class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()  # Not called, as overrided by grandchild

    def r(self, next_state, state, action):
        raise NotImplementedError()  # Not called, as overrided by grandchild

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """

        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip  # parameterizable slip probability, initially 0.1

        n_states = self.lake.size + 1  # + 1 to include the absorption state
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)  # initial p
        pi[np.where(self.lake_flat == '&')[0]] = 1.0  # setting goal state p to 1

        self.absorbing_state = n_states - 1   # set to 16 - outside lake_flat, ranging from 0-15

        # Initializing environment
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

        # Up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Indices to states (coordinates), states (coordinates) to indices
        self.indices_to_states = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        self.states_to_indices = {s: i for (i, s) in enumerate(self.indices_to_states)}

        # A 3D cube storing the transition probabilities for each state s to each new state s' through each action a
        self.tp = np.zeros((self.n_states, self.n_states, self.n_actions))

        # Modifies p values from 0 to 1 where appropriate
        for state_index, state in enumerate(self.indices_to_states):
            for state_possible_index, state_possible in enumerate(self.indices_to_states):
                for action_index, action in enumerate(self.actions):

                    next_state = (state[0] + action[0], state[1] + action[1])  # simulates action and gets next state
                    next_state_index = self.states_to_indices.get(next_state)  # gets index of next state coordinates

                    # If the next state is a possible state then the transition is probable
                    if next_state_index is not None and next_state_index == state_possible_index:
                        self.tp[state_index, next_state_index, action_index] = 1.0


    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done


    def p(self, next_state, state, action):
        return self.tp(state, next_state, action)

    # def r(self, next_state, state, action):
        # TODO:

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


    def play(env):
        actions = ['w', 'a', 's', 'd']

        state = env.reset()
        env.render()

        done = False
        while not done:
            c = input('\nMove: ')
            if c not in actions:
                raise Exception('Invalid action')

            state, r, done = env.step(actions.index(c))

            env.render()
            print('Reward: {0}.'.format(r))