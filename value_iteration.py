import numpy as np


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    for _ in range(max_iterations):
        delta = 0

        for s in range(env.n_states):
            v = value[s]
            value[s] = max(
                [
                    sum(
                        [
                            env.p(s_next, s, a) *
                            (env.r(s_next, s, a) + gamma * value[s_next])
                            for s_next in range(env.n_states)
                        ]
                    )
                    for a in range(env.n_actions)
                ]
            )

            delta = max(delta, abs(v - value[s]))

        if delta < theta:
            break

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = np.argmax(
            [
                sum(
                    [
                        env.p(s_next, s, a) *
                        (env.r(s_next, s, a) + gamma * value[s_next])
                        for s_next in range(env.n_states)
                    ]
                )
                for a in range(env.n_actions)
            ]
        )

    return policy, value