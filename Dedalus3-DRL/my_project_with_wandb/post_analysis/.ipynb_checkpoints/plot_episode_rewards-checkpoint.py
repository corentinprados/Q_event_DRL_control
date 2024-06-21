# plot_episode_rewards.py

"""
Plot the mean rewards per episode during the entire training process.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Define the directory where the episode rewards file is located
log_dir = "../tmp/"

# Load the episode rewards from the .npy file
rewards_file = os.path.join(log_dir, "rewards_per_episode.npy")
if not os.path.exists(rewards_file):
    raise FileNotFoundError(f"The rewards file was not found: {rewards_file}")

# Load the rewards data
episode_rewards = np.load(rewards_file)

print(len(episode_rewards)) # to remove just a test

# Plot the mean rewards per episode
plt.figure()
plt.scatter(range(len(episode_rewards)), episode_rewards, marker='x', s=5)  # Scatter plot for mean rewards per episode
plt.xlabel('Episode')  # Label for the x-axis
plt.ylabel('Mean Reward')  # Label for the y-axis
plt.title('Mean Rewards Per Episode Over Time')  # Title of the plot
plt.show()  # Display the plot
