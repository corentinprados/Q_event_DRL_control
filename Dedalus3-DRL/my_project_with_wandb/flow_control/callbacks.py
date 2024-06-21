# callbacks.py

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import logging
import wandb

def append_or_create_npy(file_path, new_data):
    """
    Appends new data to an existing .npy file, or creates the file if it doesn't exist.

    Parameters:
    file_path (str): The path to the .npy file.
    new_data (np.ndarray): The new data to append to the file.

    If the file at file_path exists, this function loads the existing data,
    appends the new data to it, and then saves the combined data back to the file.
    If the file does not exist, it simply saves the new data to a new file.
    """
    if os.path.exists(file_path):
        # Load existing data from the file
        existing_data = np.load(file_path)
        # Concatenate existing data with new data
        combined_data = np.concatenate((existing_data, new_data), axis=0)
        # Save the combined data back to the file
        np.save(file_path, combined_data)
    else:
        # If the file does not exist, save the new data to a new file
        np.save(file_path, new_data)

class WandbLoggingCallback(BaseCallback):
    """
    Callback for logging metrics to Wandb and saving the best model based on training reward.
    """
    def __init__(self, check_freq: int, log_dir: str, total_timesteps: int, verbose: int = 1):
        """
        Initialize the callback.

        :param check_freq: Frequency to check for saving the best model.
        :param log_dir: Directory to save the best model.
        :param total_timesteps: Total timesteps of the training.
        :param verbose: Verbosity level. Default is 1.
        """
        super(WandbLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq  # Frequency to check for saving the best model
        self.log_dir = log_dir  # Directory to save the best model
        self.total_timesteps = total_timesteps  # Total number of timesteps for training 
        self.save_path = os.path.join(log_dir, 'best_model.zip')  # Path to save the best model
        self.save_path_checkpoint = os.path.join(log_dir, 'checkpoint_model.zip')  # Path to save a checkpoint in case of the program crashes
        self.best_mean_reward = -np.inf  # Initialize best mean reward
        self.rewards_log = []  # List to store rewards
        self.actions_log = []  # List to store actions
        self.value_loss_log = []  # List to store value loss
        self.policy_loss_log = []  # List to store policy loss
        
        self.episode_rewards = []  # List to store mean rewards per episode
        self.episode_reward = 0  # Reward accumulator for the current episode
        self.episode_length = 0  # Length of the current episode

        # Configure the logger
        self._logger = logging.getLogger(__name__)
        self._logger.info("WandbLoggingCallback initialized")

    def _init_callback(self) -> None:
        """
        Initialize the callback by creating the log directory if it does not exist.
        """
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        self._logger.info("Log directory initialized")

    def _on_step(self) -> bool:
        """
        This function will be called by the model after each call to `env.step()`.
        """
        # Fetch the rewards for the current training step
        reward = np.sum(self.locals['rewards'])  # Sum of rewards for the current step
        self.rewards_log.append(reward)  # Append the reward to the rewards log

        # Log actions, value loss, and policy loss
        self.actions_log.append(self.locals['actions'])

        self.episode_reward += reward  # Accumulate the reward to compute mean reward per episode
        self.episode_length += 1  # Increment the episode length

        # Check if the episode is done, if it is the mean reward per episode is saved 
        if self.locals['dones'][0]:  
            mean_reward = self.episode_reward / self.episode_length  # Calculate the mean reward for the episode
            self.episode_rewards.append(mean_reward)  # Store the mean reward
            self.episode_reward = 0  # Reset the reward accumulator
            self.episode_length = 0  # Reset the episode length
            
            # Save the mean rewards per episode
            file_path = os.path.join(self.log_dir, "rewards_per_episode.npy")
            new_data = self.episode_rewards
            append_or_create_npy(file_path, new_data)
            wandb.log({"mean_reward_per_episode": mean_reward, "episode_length": self.episode_length})

        # Save the rewards log periodically
        if self.n_calls % self.check_freq == 0:  # Check if the current step is a multiple of check_freq

            # Make a checkpoint of the model if it crashes
            self.model.save(self.save_path_checkpoint)
            
            # Fetch the last 1000 rewards
            nb_steps_for_mean = 100
            rewards = np.array(self.rewards_log[-nb_steps_for_mean:])  # Get the last 100 rewards
            mean_reward = np.mean(rewards)  # Calculate the mean reward

            if self.verbose > 0:
                self._logger.info(f"Progress: {self.num_timesteps}/{self.total_timesteps} timesteps")
                self._logger.info(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward for the last {nb_steps_for_mean} steps: {mean_reward:.2f}\n")
                wandb.log({
                    "mean_reward_last_1000_steps": mean_reward,
                })

            # Check if the mean reward is better than the best mean reward
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward  # Update the best mean reward
                if self.verbose > 0:
                    self._logger.info(f"Saving new best model to {self.save_path}\n")
                    wandb.log({"best_model_saved": True})

                self.model.save(self.save_path)  # Save the model

            # Log statistics for the last 100 steps
            last_100_actions = np.concatenate(self.actions_log[-nb_steps_for_mean:])

            wandb.log({
                "action_mean_last_100": np.mean(last_100_actions),
                "action_std_last_100": np.std(last_100_actions),
            })
        
        return True  # Continue training
