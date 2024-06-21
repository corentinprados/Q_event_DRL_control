# plot_training_rewards.py

"""
Plot the rewards during the entire training process.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Define the directory where the training rewards file is located
log_dir = "../tmp/process_2/"


# Load the training rewards from the .npy file
rewards_file = os.path.join(log_dir, "training_rewards.npy")
if not os.path.exists(rewards_file):
    raise FileNotFoundError(f"The rewards file was not found: {rewards_file}")

# Load the rewards data
training_rewards = np.load(rewards_file)

print(len(training_rewards)) # to remove just a test

# Plot the training rewards
plt.figure()
plt.scatter(range(len(training_rewards)), training_rewards, marker='x', s=5)
plt.xlabel('Training Step')  # Label for the x-axis
plt.ylabel('Reward')  # Label for the y-axis
plt.title('Training Rewards Over Time')  # Title of the plot
plt.show()  # Display the plot
