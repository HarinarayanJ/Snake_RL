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

        reward = 0

        if (head_x < 0 or head_x >= WIDTH or head_y < 0 or head_y >= HEIGHT 
            or new_head in self.snake):
            self.done = True
            reward = -1
            return self.get_state(), reward, self.done, {}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            reward = +1
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

        danger_front = int([head_x + dx*CELL_SIZE, head_y + dy*CELL_SIZE] in self.snake or 
                           head_x + dx*CELL_SIZE < 0 or head_x + dx*CELL_SIZE >= WIDTH or 
                           head_y + dy*CELL_SIZE < 0 or head_y + dy*CELL_SIZE >= HEIGHT)
        danger_left = int([head_x - dy*CELL_SIZE, head_y + dx*CELL_SIZE] in self.snake or 
                          head_x - dy*CELL_SIZE < 0 or head_x - dy*CELL_SIZE >= WIDTH or 
                          head_y + dx*CELL_SIZE < 0 or head_y + dx*CELL_SIZE >= HEIGHT)
        danger_right = int([head_x + dy*CELL_SIZE, head_y - dx*CELL_SIZE] in self.snake or 
                           head_x + dy*CELL_SIZE < 0 or head_x + dy*CELL_SIZE >= WIDTH or 
                           head_y - dx*CELL_SIZE < 0 or head_y - dx*CELL_SIZE >= HEIGHT)

        return (danger_front, danger_left, danger_right, 
                int(food_dx < 0), int(food_dx > 0),
                int(food_dy < 0), int(food_dy > 0),
                int(self.direction=="UP"), int(self.direction=="DOWN"),
                int(self.direction=="LEFT"), int(self.direction=="RIGHT"))


from agent import ActorCriticAgent

EPISODES = 50000
PRINT_EVERY = 500

def train_actor_critic():
    env = SnakeEnv()
    agent = ActorCriticAgent(state_size=11, action_size=4, 
                             lr_actor=0.001, lr_critic=0.01, gamma=0.9)

    rewards = []
    episode_lengths = []  
    episode_steps = []    

    max_reward = -9999

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        rewards.append(total_reward)
        episode_steps.append(steps)
        episode_lengths.append(len(env.snake))

        max_reward = max(max_reward, total_reward)

        if ep % PRINT_EVERY == 0:
            avg = np.mean(rewards[-PRINT_EVERY:])
            avg_steps = np.mean(episode_steps[-PRINT_EVERY:])
            avg_length = np.mean(episode_lengths[-PRINT_EVERY:])
            print(f"Actor-Critic {ep}/{EPISODES} | avg_reward={avg:.2f} | "
                  f"avg_steps={avg_steps:.1f} | avg_length={avg_length:.1f} | best={max_reward}")

    metrics = {
        "rewards": rewards,
        "steps": episode_steps,
        "lengths": episode_lengths
    }
    pickle.dump(metrics, open("actor_critic_training_log.pkl", "wb"))
    
    agent_params = {
        "theta": agent.theta,
        "w": agent.w,
        "state_size": agent.state_size,
        "action_size": agent.action_size
    }
    pickle.dump(agent_params, open("actor_critic_params.pkl", "wb"))

    print("\nActor-Critic Training complete!")
    print("Best episode reward:", max_reward)
    print("Metrics saved to actor_critic_training_log.pkl")
    print("Agent parameters saved to actor_critic_params.pkl")


if __name__ == "__main__":
    train_actor_critic()