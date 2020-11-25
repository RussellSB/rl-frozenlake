from frozen_lake import FrozenLake
from policy_iteration import policy_iteration
from value_iteration import value_iteration
from tabular_model_free import sarsa, q_learning

def main():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    #env.play()

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 10000

    #print('')

    #print('## Policy iteration')
    #policy, value = policy_iteration(env, gamma, theta, max_iterations)
    #env.render(policy, value)

    #print('')

    #print('## Value iteration')
    #policy, value = value_iteration(env, gamma, theta, max_iterations)
    #env.render(policy, value)

    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print('## sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon)
    #print('## q_learning')
    #policy, value = q_learning(env, max_episodes, eta, gamma, epsilon)
    env.render(policy, value)

main()