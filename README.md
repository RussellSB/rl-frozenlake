# reinforcement-learning
A variety of reinforcement learning algorithms applied to the scenario of a grid world variant.

This variant is a frozen lake environement:
- One square denotes a goal state
- Some squares denote holes (trap states)
- The rest is ice. Plain ground for the agent to walk on.

After reaching the goal state, the agent recieves a reward when commiting an action from it. From the goal state, the agent gets placed into an absorption state, where every action they take there loops back to the same state and they cannot leave. The same logic is similar for holes, except that the agent recieves no reward when moving to the absorption state. Furthermore the ice states are slippery. On performing an action on them there is a 10% probabiliy that the agent may slip and move in a random direction. Possible actions are up, left, right, down.

The agent may move for a number of steps. In this period the reward can either conclude to 1 (goal reached) or 0 (goal not reached).

## Implemented
- Policy iteration
- Frozen lake environment
- Value iteration
- Sarsa control
- Q-learning
- Sara control + linear function approximation
- Q-learning control + linear function approximation

## How to run
You may run any of the above algorithms through interpreting the main file main.py. In here, all methods with corresponding names for the algorithms may be found. The environment on which the algorithms are ran on may be redifined as small_lake or big_lake. Feel free to redefine it to an entirely new environment as well, as long as its in a list format with $, ., #, and & characters for goal, ice, hole, and agent states respectively.
