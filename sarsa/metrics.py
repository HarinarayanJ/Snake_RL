import pickle
import matplotlib.pyplot as plt

with open("sarsa_training_log.pkl", "rb") as f:
    data = pickle.load(f)

rewards = data['rewards']
lengths = data['lengths']

episodes = list(range(len(rewards)))

plt.figure(figsize=(10, 5))
plt.plot(episodes, lengths, label='Episode Length', color='lightblue')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.title('Episode vs Length')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards, label='Episode Reward', color='lightblue')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode vs Reward')
plt.grid(True)
plt.legend()
plt.show()
