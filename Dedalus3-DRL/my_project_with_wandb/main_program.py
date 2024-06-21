# main_program.py

import os
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from flow_control.flow_control_env import FlowControlEnv
from flow_control.callbacks import SaveOnBestTrainingRewardCallback
from joblib import dump, load
from logging_setup import setup_logging
import torch.nn as nn

def print_model_architecture(model):
    """
    Print the architecture of the policy network of the PPO model.
    """
    print("Model Architecture:")
    print(model.policy)

def train_environment(restart=False):
    """
    Train or continue training a PPO model for the flow control environment.

    Parameters:
    restart (bool): If True, training starts from scratch even if a pre-trained model exists.
                    If False, continues training from the existing model if available.

    This function sets up logging, initializes the environment and model,
    and trains the model while saving the best version and the environment state.
    """
    # Directory for logging and saving models
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    print("Setting up logging...\n")
    setup_logging(log_dir, log_filename="training.log")
    print("Logging setup complete.\n")

    logger = logging.getLogger(__name__)
    logger.info("Starting the training environment\n")

    # Training parameters
    n_steps = 4096
    nb_times_nstep = 20
    total_timesteps = nb_times_nstep * n_steps
    check_freq = n_steps // 4

    # Initialize the environment
    env = FlowControlEnv()
    
    # Path to the model
    model_path = os.path.join(log_dir, "ppo_flow_control_trained.zip")

    if not restart and os.path.exists(model_path):
        # Load existing model
        model = PPO.load(model_path, env=env)
        logger.info("Loaded existing model for further training\n")
    else:
        # Train from scratch
        model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0001, n_steps=n_steps, batch_size=128, ent_coef = 0.01)
        if restart:
            logger.info("Restarting training from scratch\n")
        else:
            logger.info("No existing model found, starting training from scratch\n")

    # Print the model architecture
    print_model_architecture(model)

    # Create the callback for saving the best model and logging progress
    save_callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir, total_timesteps=total_timesteps)

    try:
        # Train the agent
        logger.info("Starting model training\n")
        model.learn(total_timesteps=total_timesteps, callback=[save_callback])

        # Save the model
        model.save(model_path)
        logger.info("Model saved successfully\n")

        # Save the environment state using joblib
        saveable_state = env.get_saveable_state()
        dump(saveable_state, os.path.join(log_dir, "flow_control_env_state.joblib"))
        logger.info("Environment state saved successfully\n")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    restart = False  # Set this to True to restart training from scratch
    train_environment(restart=restart)
