import numpy as np

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    # Iterate until the max iteration
    for _ in range(max_iterations):

        delta = 0
        for s in range(env.n_states):

            v = value[s]

            # Computing the current value for policy evaluation
            value[s] = sum(
                [
                    env.p(next_s, s, policy[s]) *
                    ((env.r(next_s, s, policy[s]) + gamma * value[next_s]))
                    for next_s in range(env.n_states)
                ]
            )

            delta = max(delta, abs(v - value[s]))  # difference to check convergence

        # Breaks when policy converges
        if delta < theta:
            break
    return value

def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    for s in range(env.n_states):
        policy[s] = np.argmax(  # picks action for each state with the highest expected reward
            [
                sum(
                    [
                        env.p(next_s, s, a) *
                        ((env.r(next_s, s, a) + gamma * value[next_s]))
                        for next_s in range(env.n_states)
                    ]
                )
                for a in range(env.n_actions)
            ]
        )

    # for s in range(env.n_states):
    #     actions_to_max = []
    #     for a in range(env.n_actions):
    #         term = 0
    #         for next_s in range(env.n_states):
    #             term += env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s])
    #
    #         actions_to_max.append(term)
    #     policy[s] = max(actions_to_max)

    return policy

def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    while(True):
        policy_initial = policy

        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)

        if np.array_equal(policy_initial, policy):
            break

    return policy, value
