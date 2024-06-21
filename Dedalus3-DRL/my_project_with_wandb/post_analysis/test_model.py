# test_model.py

"""
Test the saved model
"""

import os
import sys
import numpy as np
from joblib import load
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Add the parent directory to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FlowControlEnv from the flow_control package
from flow_control.flow_control_env import FlowControlEnv

log_dir = "../tmp/process_5/"

# Load the trained model
model = PPO.load(os.path.join(log_dir, "best_model.zip"))

# Load the saved environment state using joblib
state_path = os.path.join(log_dir, "flow_control_env_state.joblib")
if not os.path.exists(state_path):
    raise FileNotFoundError(f"The environment state file was not found: {state_path}")

saveable_state = load(state_path)

# Initialize the environment and set its state
env = FlowControlEnv()
env.set_saveable_state(saveable_state)

# Test the trained model
obs, _ = env.reset()
rewards = []
actions = []
agent_steps = []

def plot_action_reward(actions, rewards, simulation_steps, agent_steps):
    """
    Visualizes the rewards and actions over time using plt.pcolormesh.

    Parameters:
    actions (list): The actions taken by the agent.
    rewards (list): The rewards received over time.
    simulation_steps (ndarray): The simulation step indices.
    agent_steps (list): The steps at which actions were taken.
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Plot rewards over time
    axes[0].plot(simulation_steps, rewards)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_title('Rewards Over Time')
    axes[0].set_aspect('auto')

    # Plot actions over time
    axes[1].scatter(agent_steps, actions)
    axes[1].set_title('Actions Over Time')
    axes[1].set_aspect('auto')

def print_model_architecture(model):
    """
    Print the architecture of the policy network of the PPO model.
    """
    print("Model Architecture:")
    print(model.policy)

# Print the model architecture
print_model_architecture(model)

for step in range(10000):  # Run for a certain number of steps to test
    action, _states = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    rewards.append(reward)
    state = env.get_saveable_state()
    delta_t = state['delta_t']

    # Check if the agent action was actually taken
    if delta_t > 0:
        actions.append(action[0])
        agent_steps.append(step)

    # Render the environment at regular intervals
    if step % 100 == 0 and step != 0:
        env.render()
        plt.pause(1)  # Pause to allow the figure to be displayed
        input("Press Enter to continue...")  # Control pause for user input
        plt.close()

    # Plot the rewards and actions at regular intervals
    if step % 1000 == 0 and step != 0:
        simulation_steps = np.arange(step + 1)
        plot_action_reward(actions, rewards, simulation_steps, agent_steps)
        plt.pause(1)  # Pause to allow the figure to be displayed
        input("Press Enter to continue...")  # Control pause for user input
        plt.close()

    if done:
        # Exit the loop if the episode is done
        break
