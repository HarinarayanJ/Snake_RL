import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

CELL_SIZE = 20
WIDTH, HEIGHT = 600, 400
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE

class SnakeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = "RIGHT"
        self.snake = [[5, 5]]
        self.score = 0
        self.done = False

        self._place_food()
        return self._get_state()

    def _place_food(self):
        self.food = [
            random.randint(0, COLS - 1),
            random.randint(0, ROWS - 1)
        ]
        if self.food in self.snake:
            self._place_food()

    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Basic state representation
        return np.array([
            head_x, head_y,
            food_x, food_y,
            int(self.direction == "UP"),
            int(self.direction == "DOWN"),
            int(self.direction == "LEFT"),
            int(self.direction == "RIGHT"),
        ], dtype=np.float32)

    def step(self, action):
        # 0 = up, 1 = down, 2 = left, 3 = right
        if action == 0 and self.direction != "DOWN": self.direction = "UP"
        elif action == 1 and self.direction != "UP": self.direction = "DOWN"
        elif action == 2 and self.direction != "RIGHT": self.direction = "LEFT"
        elif action == 3 and self.direction != "LEFT": self.direction = "RIGHT"

        x, y = self.snake[0]
        if self.direction == "UP":    y -= 1
        if self.direction == "DOWN":  y += 1
        if self.direction == "LEFT":  x -= 1
        if self.direction == "RIGHT": x += 1

        new_head = [x, y]

        # Check collisions (death)
        done = (
            x < 0 or x >= COLS or
            y < 0 or y >= ROWS or
            new_head in self.snake
        )

        reward = -1  # small penalty per move

        if done:
            reward = -100  # death penalty
            self.done = True
        else:
            self.snake.insert(0, new_head)

            if new_head == self.food:
                reward = 50  # reward for eating food
                self.score += 1
                self._place_food()
            else:
                self.snake.pop()

        return self._get_state(), reward, done, {}



# --- Neural Network for Q-values ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.update_target()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return int(torch.argmax(q_values).item())

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch

        states      = torch.tensor(states, dtype=torch.float32)
        actions     = torch.tensor(actions)
        rewards     = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones       = torch.tensor(dones, dtype=torch.float32)

        # Compute current Q-values
        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values
        next_q = self.target_model(next_states).max(1)[0]
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(q_value, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
