import random
import pickle
import numpy as np

CELL_SIZE = 20
WIDTH, HEIGHT = 600, 400

ACTIONS = [0, 1, 2, 3]  # 0 for UP,1 for DOWN,2 for LEFT,3 for RIGHT

class SnakeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [[100, 100]]
        self.direction = "RIGHT"
        self.food = [random.randrange(0, WIDTH, CELL_SIZE),
                     random.randrange(0, HEIGHT, CELL_SIZE)]
        self.done = False
        return self.get_state()

    def step(self, action):
        if action == 0 and self.direction != "DOWN": self.direction = "UP"
        elif action == 1 and self.direction != "UP": self.direction = "DOWN"
        elif action == 2 and self.direction != "RIGHT": self.direction = "LEFT"
        elif action == 3 and self.direction != "LEFT": self.direction = "RIGHT"

        head_x, head_y = self.snake[0]
        if self.direction == "UP": head_y -= CELL_SIZE
        elif self.direction == "DOWN": head_y += CELL_SIZE
        elif self.direction == "LEFT": head_x -= CELL_SIZE
        elif self.direction == "RIGHT": head_x += CELL_SIZE

        new_head = [head_x, head_y]

        reward = -1  # move penalty to reduce looping

        # Check death
        if (head_x < 0 or head_x >= WIDTH or head_y < 0 or head_y >= HEIGHT or new_head in self.snake):
            self.done = True
            reward = -100 #dont want to die, so penality for dying
            return self.get_state(), reward, self.done, {}

        self.snake.insert(0, new_head)

        # Check food
        if new_head == self.food:
            reward = 50
            while True:  # spawn food not on snake
                self.food = [random.randrange(0, WIDTH, CELL_SIZE),
                             random.randrange(0, HEIGHT, CELL_SIZE)]
                if self.food not in self.snake:
                    break
        else:
            self.snake.pop()

        return self.get_state(), reward, self.done, {}

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_dx = (self.food[0] - head_x) // CELL_SIZE
        food_dy = (self.food[1] - head_y) // CELL_SIZE

        dx, dy = 0, 0
        if self.direction == "UP": dx, dy = 0, -1
        elif self.direction == "DOWN": dx, dy = 0, 1
        elif self.direction == "LEFT": dx, dy = -1, 0
        elif self.direction == "RIGHT": dx, dy = 1, 0

        danger_front = int([head_x + dx, head_y + dy] in self.snake or 
                           head_x + dx < 0 or head_x + dx >= WIDTH or 
                           head_y + dy < 0 or head_y + dy >= HEIGHT)
        danger_left = int([head_x - dy, head_y + dx] in self.snake or 
                          head_x - dy < 0 or head_x - dy >= WIDTH or 
                          head_y + dx < 0 or head_y + dx >= HEIGHT)
        danger_right = int([head_x + dy, head_y - dx] in self.snake or 
                           head_x + dy < 0 or head_x + dy >= WIDTH or 
                           head_y - dx < 0 or head_y - dx >= HEIGHT)

        return (danger_front, danger_left, danger_right, 
                int(food_dx < 0), int(food_dx > 0),
                int(food_dy < 0), int(food_dy > 0),
                int(self.direction=="UP"), int(self.direction=="DOWN"),
                int(self.direction=="LEFT"), int(self.direction=="RIGHT"))


class QLearningAgent:
    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.1):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}  # Q-table

    def get_action(self, state):
        if state not in self.Q:
            self.Q[state] = [0] * 4
        if np.random.rand() < self.epsilon:
            return random.choice([0,1,2,3])
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state):
        if state not in self.Q:
            self.Q[state] = [0] * 4
        if next_state not in self.Q:
            self.Q[next_state] = [0] * 4
        self.Q[state][action] += self.lr * (reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action])


EPISODES = 75000
PRINT_EVERY = 500

def train_q_learning():
    env = SnakeEnv()
    agent = QLearningAgent(lr=0.1, gamma=0.9, epsilon=0.1)
    rewards = []
    max_reward = -9999

    for ep in range(1, EPISODES + 1):
        s = env.reset()
        total_reward = 0
        while True:
            a = agent.get_action(s)
            ns, r, done, _ = env.step(a)
            agent.update(s, a, r, ns)
            s = ns
            total_reward += r
            if done:
                break
        rewards.append(total_reward)
        max_reward = max(max_reward, total_reward)
        if ep % PRINT_EVERY == 0:
            avg = np.mean(rewards[-PRINT_EVERY:])
            print(f"Episode {ep}/{EPISODES} | avg_reward={avg:.2f} | best={max_reward}")

    pickle.dump(agent.Q, open("qtable.pkl", "wb"))
    print("\nTraining complete! Best episode reward:", max_reward)

if __name__ == "__main__":
    train_q_learning()
