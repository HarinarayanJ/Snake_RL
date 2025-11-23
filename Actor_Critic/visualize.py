import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("actor_critic_training_log.pkl", "rb") as f:
    data = pickle.load(f)

rewards = data['rewards']
lengths = data['lengths']
steps = data['steps']
episodes = list(range(len(rewards)))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Actor-Critic Training Results', fontsize=16, fontweight='bold')

axes[0, 0].plot(episodes, rewards, alpha=0.3, color='lightblue', label='Raw Rewards')
window = 500
if len(rewards) >= window:
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 
                    color='blue', linewidth=2, label=f'Moving Avg ({window})')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].set_title('Episode vs Total Reward')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(episodes, lengths, alpha=0.3, color='lightgreen', label='Raw Length')
if len(lengths) >= window:
    moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
    axes[0, 1].plot(range(window-1, len(lengths)), moving_avg, 
                    color='green', linewidth=2, label=f'Moving Avg ({window})')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Snake Length')
axes[0, 1].set_title('Episode vs Snake Length')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[1, 0].plot(episodes, steps, alpha=0.3, color='lightsalmon', label='Raw Steps')
if len(steps) >= window:
    moving_avg = np.convolve(steps, np.ones(window)/window, mode='valid')
    axes[1, 0].plot(range(window-1, len(steps)), moving_avg, 
                    color='red', linewidth=2, label=f'Moving Avg ({window})')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Steps Survived')
axes[1, 0].set_title('Episode vs Steps Survived')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

last_1000 = min(1000, len(rewards))
stats_text = f"""
Training Statistics (Last {last_1000} episodes):

Average Reward: {np.mean(rewards[-last_1000:]):.2f}
Max Reward: {np.max(rewards):.2f}
Min Reward: {np.min(rewards):.2f}

Average Length: {np.mean(lengths[-last_1000:]):.2f}
Max Length: {np.max(lengths):.0f}

Average Steps: {np.mean(steps[-last_1000:]):.2f}
Max Steps: {np.max(steps):.0f}

Total Episodes: {len(rewards)}
"""
axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')
axes[1, 1].set_title('Training Summary')

plt.tight_layout()
plt.savefig('actor_critic_training_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'actor_critic_training_results.png'")
plt.show()

fig2, ax = plt.subplots(figsize=(12, 6))
window = 1000
if len(rewards) >= window:
    reward_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    length_ma = np.convolve(lengths, np.ones(window)/window, mode='valid')
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(range(window-1, len(rewards)), reward_ma, 
                    color='blue', linewidth=2, label='Avg Reward')
    line2 = ax2.plot(range(window-1, len(lengths)), length_ma, 
                     color='green', linewidth=2, label='Avg Length')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward', color='blue')
    ax2.set_ylabel('Average Snake Length', color='green')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    
    plt.title('Actor-Critic Learning Progress (Moving Average)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actor_critic_learning_progress.png', dpi=300, bbox_inches='tight')
    print("Learning progress saved as 'actor_critic_learning_progress.png'")
    plt.show()