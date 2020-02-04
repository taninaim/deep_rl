alpha = 0.9
gamma = 0.9
reward_table = [-0.1, 0, 1]
# -0.1 for action 0, 0 for action 1, and 1 for action 2.

q_table = [[0.1, 0.2, 0.3],
           [0.4, 0.5, 0.6],
           [0.7, 0.8, 0.9]]


def update_q(q_table, s, a, s_, a_):
    q_table[s][a] += alpha*(reward_table[a] +
                            gamma*q_table[s_][a_] - q_table[s][a])
    return q_table


if __name__ == "__main__":
    s = 0
    a = 0
    s_ = 1
    a_ = 1
    print(q_table)
    q_table = update_q(q_table, s, a, s_, a_)
    print(q_table)
