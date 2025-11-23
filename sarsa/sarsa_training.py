import random
import pickle
import numpy as np

CELL_SIZE = 20
WIDTH, HEIGHT = 600, 400

ACTIONS = [0, 1, 2, 3]

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

        reward = -1

        # collision
        if (head_x < 0 or head_x >= WIDTH or head_y < 0 or head_y >= HEIGHT 
            or new_head in self.snake):
            self.done = True
            reward = -100
            return self.get_state(), reward, self.done, {}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            reward = 50
            while True:
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

from sarsa_agent import SARSAAgent

EPISODES = 200000
PRINT_EVERY = 500

def train_sarsa():
    env = SnakeEnv()
    agent = SARSAAgent(lr=0.1, gamma=0.9, epsilon=0.1)

    rewards = []
    episode_lengths = []  # snake length per episode
    episode_steps = []    # steps survived per episode

    max_reward = -9999

    for ep in range(1, EPISODES + 1):
        s = env.reset()
        a = agent.get_action(s)
        total_reward = 0
        steps = 0

        while True:
            ns, r, done, _ = env.step(a)
            na = agent.get_action(ns)

            agent.update(s, a, r, ns, na)

            s, a = ns, na
            total_reward += r
            steps += 1

            if done:
                break

        rewards.append(total_reward)
        episode_steps.append(steps)
        episode_lengths.append(len(env.snake))

        max_reward = max(max_reward, total_reward)

        if ep % PRINT_EVERY == 0:
            avg = np.mean(rewards[-PRINT_EVERY:])
            print(f"SARSA {ep}/{EPISODES} | avg={avg:.2f} | best={max_reward}")

    metrics = {
        "rewards": rewards,
        "steps": episode_steps,
        "lengths": episode_lengths
    }
    pickle.dump(metrics, open("sarsa_training_log.pkl", "wb"))
    pickle.dump(agent.Q, open("sarsa_table.pkl", "wb"))

    print("\nSARSA Training complete!")
    print("Best episode reward:", max_reward)
    print("Metrics saved to sarsa_training_log.pkl")


if __name__ == "__main__":
    train_sarsa()
