import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, lr=0.01, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.action, dtype=np.float64)

    def choose_action(self, observation):
        self.add_state(observation)
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))
            action = state_action.idxmax()
        return action

    def learn(self, s, a, r, s_):
        self.add_state(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def add_state(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions),
                          index=self.q_table.columns,
                          name=state)
            )
