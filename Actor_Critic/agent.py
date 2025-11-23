import numpy as np

class ActorCriticAgent:
    def __init__(self, state_size=11, action_size=4, lr_actor=0.001, lr_critic=0.01, gamma=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.theta = np.random.randn(state_size, action_size) * 0.01
        self.w = np.random.randn(state_size) * 0.01
    
    def get_action_probs(self, state):
        state = np.array(state, dtype=np.float32)
        logits = np.dot(state, self.theta)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
    def get_action(self, state):
        probs = self.get_action_probs(state)
        action = np.random.choice(self.action_size, p=probs)
        return action
    
    def predict_action(self, state):
        probs = self.get_action_probs(state)
        return int(np.argmax(probs))
    
    def get_value(self, state):
        state = np.array(state, dtype=np.float32)
        return np.dot(state, self.w)
    
    def update(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        
        current_value = self.get_value(state)
        next_value = 0 if done else self.get_value(next_state)
        td_error = reward + self.gamma * next_value - current_value
        
        self.w += self.lr_critic * td_error * state
        probs = self.get_action_probs(state)
        dsoftmax = probs.copy()
        dsoftmax[action] -= 1  
        
        for a in range(self.action_size):
            self.theta[:, a] -= self.lr_actor * td_error * dsoftmax[a] * state