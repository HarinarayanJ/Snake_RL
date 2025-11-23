import numpy as np

class SARSAAgent:
    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.1):
        self.Q = {}
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state):
        state = tuple(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 4)
        return self.predict_action(state)

    def predict_action(self, state):
        state = tuple(state)
        if state not in self.Q:
            self.Q[state] = [0, 0, 0, 0]
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, ns, na):
        s = tuple(s)
        ns = tuple(ns)

        if s not in self.Q:
            self.Q[s] = [0, 0, 0, 0]
        if ns not in self.Q:
            self.Q[ns] = [0, 0, 0, 0]

        td_target = r + self.gamma * self.Q[ns][na]
        td_error  = td_target - self.Q[s][a]

        self.Q[s][a] += self.lr * td_error
