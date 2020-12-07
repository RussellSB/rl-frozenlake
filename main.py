from frozen_lake import FrozenLake
from policy_iteration import policy_iteration
from value_iteration import value_iteration
from tabular_model_free import sarsa, q_learning
import numpy as np
from non_tabular_model_free import linear_q_learning, linear_sarsa, LinearWrapper


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

    # lake = big_lake
    lake = small_lake
    size = len(lake) * len(lake[0])
    env = FrozenLake(lake, slip=0.1, max_steps=size, seed=seed)
    #env.play()

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
    optimal_policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(optimal_policy, value)

    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5
    print('# Model-free algorithms')
    print('## sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    print('## q_learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('# Model-free algorithms')
    linear_env = LinearWrapper(env)

    print('## linear sarsa')
    parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('## linear q_learning')
    parameters = linear_q_learning(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('# finding number of episodes for optimal policy')
    for episodes in np.arange(500,5000,100):
        print(f'sarsa episodes = {episodes}')
        policy, value = sarsa(env, episodes, eta, gamma, epsilon, seed=seed)
        policy[15] = 0  # set the policy for the goal state as 0 to compare with optimal policy
        if np.array_equal(policy, optimal_policy):
            break
    env.render(policy, value)

    for episodes in np.arange(500,5000,100):
        print(f'q_learning episodes = {episodes}')
        policy, value = q_learning(env, episodes, eta, gamma, epsilon, seed=seed)
        policy[15] = 0  # set the policy for the goal state as 0 to compare with optimal policy
        if np.array_equal(policy, optimal_policy):
            break
    env.render(policy, value)

    print('find best policy on big map')

    lake = big_lake
    size = len(lake) * len(lake[0])
    env = FrozenLake(lake, slip=0.1, max_steps=size, seed=seed)

    print('## Value iteration')
    optimal_policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(optimal_policy, value)

    max_episodes = 11000
    eta = 0.5
    epsilon = 0.99
    gamma = 0.91

    linear_env = LinearWrapper(env)

    print('## linear sarsa')
    parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    dif = (policy == optimal_policy).sum()
    print(f'sarsa difference to optimal = {100*(dif)/size}%')

    print('## linear q_learning')
    parameters = linear_q_learning(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    dif = (policy == optimal_policy).sum()
    print(f'q_learning difference to optimal = {100*(dif)/size}%')
main()
