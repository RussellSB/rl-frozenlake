from frozen_lake import FrozenLake
from policy_iteration import policy_iteration
from value_iteration import value_iteration
from tabular_model_free import sarsa, q_learning
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

    # print('## Policy iteration')
    # policy, value = policy_iteration(env, gamma, theta, max_iterations)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Value iteration')
    # policy, value = value_iteration(env, gamma, theta, max_iterations)
    # env.render(policy, value)

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


main()
