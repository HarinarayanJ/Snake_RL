from dqn_agent import SnakeEnv, DQNAgent
from replay_buffer import ReplayBuffer
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import os

EPISODES = 10000
BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 50
PLOT_EVERY = 1  # update plot every episode

def train_dqn():
    env = SnakeEnv()
    state_dim = len(env.reset())
    action_dim = 4

    agent = DQNAgent(state_dim, action_dim)
    buffer = ReplayBuffer(20000)

    rewards = []
    avg_rewards = []
    lengths = []
    avg_lengths = []

    # Setup live plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        snake_length = 1  # initial snake length

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Update snake length
            snake_length = max(snake_length, len(env.snake))

            if len(buffer) > BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                agent.train_step(batch)

            if done:
                break

        rewards.append(total_reward)
        lengths.append(snake_length)  # store snake length instead of steps

        avg_rewards.append(np.mean(rewards[-50:]))
        avg_lengths.append(np.mean(lengths[-50:]))

        # Sync target network periodically
        if ep % UPDATE_TARGET_EVERY == 0:
            agent.update_target()

        print(f"Episode {ep} | Reward: {total_reward} | Avg Reward: {avg_rewards[-1]:.2f} | Snake Length: {snake_length} | Avg Length: {avg_lengths[-1]:.2f}")

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

            ax2.plot(lengths, label='Snake Length')
            ax2.plot(avg_lengths, label='Average Length (50 ep)')
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Snake Length")
            ax2.set_title("Episode vs Snake Length")
            ax2.legend()

            plt.pause(0.01)


    plt.ioff()
    plt.show()

    plt.close('all') 

    # Save trained model
    
    torch.save(agent.model.state_dict(), "dqn_snake.pth")
    print("Training done! Model saved as 'dqn_snake.pth'")

if __name__ == "__main__":
    train_dqn()
