from frozen_lake import FrozenLake
from policy_iteration import policy_iteration
from value_iteration import value_iteration
from tabular_model_free import sarsa, q_learning
from non_tabular_model_free import linear_q_learning, linear_sarsa, LinearWrapper
import numpy as np

def main():
    seed = 0

    # Small lake
    small_lake =    [['&', '.', '.', '.'],
                    ['.', '#', '.', '#'],
                    ['.', '.', '.', '#'],
                    ['#', '.', '.', '$']]

    big_lake =      [['&', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '#', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '#', '.', '.'],
                    ['.', '.', '.', '#', '.', '.', '.', '.'],
                    ['.', '#', '#', '.', '.', '.', '#', '.'],
                    ['.', '#', '.', '.', '#', '.', '#', '.'],
                    ['.', '.', '.', '#', '.', '.', '.', '$']]

    env = FrozenLake(small_lake, slip=0.1, max_steps=16, seed=seed)
    # env.play()

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 10000

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5
    sarsa_policy = policy * 0
    q_learning_policy = policy * 0
    linear_sarsa_policy = 0
    linear_q_learning_policy = 0
    for episodes in np.arange(100,4000,100):
        print(f'episodes = {episodes}')
        if not (np.array_equal(sarsa_policy, policy)):
            sarsa_policy, value = sarsa(env, episodes, eta, gamma, epsilon)
            if np.array_equal(sarsa_policy,policy):
                print(f"sarsa optimal = {episodes}")
                env.render(sarsa_policy, value)

        if not np.array_equal(q_learning_policy, policy):
            q_learning_policy, value = sarsa(env, episodes, eta, gamma, epsilon)
            if np.array_equal(q_learning_policy, policy):
                print(f"q_learning_policy optimal = {episodes}")
                env.render(q_learning_policy, value)

    print('sarsa final')
    env.render(sarsa_policy, value)
    print('q learning final')
    env.render(q_learning_policy, value)

    # print('# Model-free algorithms')
    # linear_env = LinearWrapper(env)
    #
    # print('## linear sarsa')
    # parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)
    #
    # print('## linear q_learning')
    # parameters = linear_q_learning(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)


main()
