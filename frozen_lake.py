import numpy as np
from itertools import product
import contextlib

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()  # Overrided by FrozenLake

    def r(self, next_state, state, action):
        raise NotImplementedError()  # Overrided by FrozenLake

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        # Scenario action 3 (right) - where slip is 0.2
        # Moving from state 0. We want an array like this:
        # p = [0.1, 0.8, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # print(p)
        next_state = self.random_state.choice(self.n_states, p=p)  # chooses state with highest
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
        raise NotImplementedError()  # Overrided by FrozenLake


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
        pi[np.where(self.lake_flat == '&')[0]] = 1.0  # setting start state p to 1

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

        # Models environment deterministically
        # Modifies p values from 0 to 1 where appropriate
        for state_index, state in enumerate(self.indices_to_states):
            for state_possible_index, state_possible in enumerate(self.indices_to_states):
                for action_index, action in enumerate(self.actions):

                    # Checks if hole or goal, to only enable absorption state transitions
                    state_char = self.lake_flat[state_index]
                    if state_char == '$' or state_char == '#':
                        self.tp[state_index, n_states-1, action_index] = 1.0
                        continue

                    # Proceeds normally

                    next_state = (state[0] + action[0], state[1] + action[1])  # simulates action and gets next state
                    next_state_index = self.states_to_indices.get(next_state)  # gets index of next state coordinates

                    # If the next state is a possible state then the transition is probable
                    if next_state_index is not None and next_state_index == state_possible_index:
                        self.tp[state_index, next_state_index, action_index] = 1.0

                    # If next_state is out of bounds, default next state to current state index
                    if next_state_index is None:
                        next_state_index = self.states_to_indices.get(next_state, state_index)
                        self.tp[state_index, next_state_index, action_index] = 1.0

            # Remodels each state-state-action array to cater for slipping
            valid_states, valid_actions = np.where(self.tp[state_index] == 1)
            valid_states = np.unique(valid_states)  # At borders can have actions that map to the same state

            for state_possible_index, state_possible in enumerate(self.indices_to_states):
                for action_index, action in enumerate(self.actions):

                    # Readjust the p=1 value so that it distributes along side the slipping probabilities
                    if self.tp[state_index, state_possible_index, action_index] == 1:
                        self.tp[state_index, state_possible_index, action_index] -= self.slip

                    # if the state is reachable with other actions (hence 0), and if the action exists
                    if self.tp[state_index, state_possible_index, action_index] == 0 and \
                            state_possible_index in valid_states and action_index in valid_actions:
                        # Change p from 0 to a probability determined by slip and valid states (excluding the p=1 one)
                        self.tp[state_index, state_possible_index, action_index] = self.slip / (len(valid_states)-1)



    def step(self, action):
        state, reward, done = Environment.step(self, action)  # else, transition normally
        done = (state == self.absorbing_state) or done
        return state, reward, done


    def p(self, next_state, state, action):
        return self.tp[state, next_state, action]

    def r(self, next_state, state, action):
        char = 'o'

        # if within env boundaries
        if(state < self.n_states-1):
            char = self.lake_flat[state]  # get char of state in environment

        if(char == '$'): # If moving from goal state
            return 1

        return 0


    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['↑', '↓', '←', '→']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


    def play(self):
        actions = ['w', 's', 'a', 'd']

        state = self.reset()
        self.render()

        done = False
        while not done:
            c = input('\nMove: ')
            if c not in actions:
                raise Exception('Invalid action')

            state, r, done = self.step(actions.index(c))

            self.render()
            print('Reward: {0}.'.format(r))

