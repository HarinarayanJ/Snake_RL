from dqn_agent import DQNAgent, SnakeEnv
from replay_buffer import ReplayBuffer
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import os

EPISODES = 10000        
BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 50
PLOT_EVERY = 1
MAX_STEPS = 5000        # maximum steps per episode

def train_dqn():
    env = SnakeEnv()
    state_dim = len(env.reset())
    action_dim = 4

    agent = DQNAgent(state_dim, action_dim)
    buffer = ReplayBuffer(20000)

    rewards = []
    avg_rewards = []
    snake_lengths = []
    avg_lengths = []

    # Setup live plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_snake_length = len(env.snake)  # track snake growth

        while True:
            steps += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            max_snake_length = max(max_snake_length, len(env.snake))

            buffer.push((state, action, reward, next_state, done))

            if len(buffer) > BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                agent.train_step(batch)

            if done or steps >= MAX_STEPS:
                break

        rewards.append(total_reward)
        snake_lengths.append(max_snake_length)
        avg_rewards.append(np.mean(rewards[-50:]))
        avg_lengths.append(np.mean(snake_lengths[-50:]))

        # Update target network periodically
        if ep % UPDATE_TARGET_EVERY == 0:
            agent.update_target()

        print(f"Episode {ep} | Reward: {total_reward} | Avg Reward: {avg_rewards[-1]:.2f} | "
              f"Max Snake Length: {max_snake_length} | Avg Snake Length: {avg_lengths[-1]:.2f}")

        # Live plot
        if ep % PLOT_EVERY == 0:
            ax1.clear()
            ax2.clear()
            ax1.plot(rewards, label='Episode Reward')
            ax1.plot(avg_rewards, label='Average Reward (50 ep)')
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.set_title("Episode vs Reward")
            ax1.legend()

            ax2.plot(snake_lengths, label='Max Snake Length')
            ax2.plot(avg_lengths, label='Average Length (50 ep)')
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Snake Length")
            ax2.set_title("Episode vs Snake Length")
            ax2.legend()

            plt.pause(0.01)

    plt.ioff()
    plt.show()
    plt.close('all')

    # Save model
    torch.save(agent.model.state_dict(), os.path.join(os.getcwd(), "dqn_snake.pth"))
    pickle.dump(rewards, open("dqn_rewards.pkl", "wb"))
    print("Training done! Model saved as 'dqn_snake.pth'")

if __name__ == "__main__":
    train_dqn()
