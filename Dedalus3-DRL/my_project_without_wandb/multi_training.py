# multi_training.py

import os
import logging
import multiprocessing
from stable_baselines3 import PPO
from flow_control.flow_control_env import FlowControlEnv 
from flow_control.callbacks import SaveOnBestTrainingRewardCallback 
from joblib import dump
from logging_setup import setup_logging  

def print_model_architecture(model):
    """
    Print the architecture of the policy network of the PPO model.
    """
    print("Model Architecture:")
    print(model.policy)

def train_environment(config, restart=False):
    """
    Train or continue training a PPO model for the flow control environment.

    Parameters:
    config (dict): A dictionary containing learning_rate, batch_size, gamma, and process_id.
    restart (bool): If True, training starts from scratch even if a pre-trained model exists.
                    If False, continues training from the existing model if available.

    This function sets up logging, initializes the environment and model,
    and trains the model while saving the best version and the environment state.
    """
    # Extract parameters from config
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    gamma = config["gamma"]
    process_id = config["process_id"]

    # Directory for logging and saving models
    log_dir = f"tmp/process_{process_id}/"
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging
    setup_logging(log_dir, log_filename="training.log")

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.info(f"Starting the training environment for process {process_id}\n")

    # Training parameters
    n_steps = 4096  # Number of steps per update
    nb_times_nstep = 20  # Number of times n_steps will be executed
    total_timesteps = nb_times_nstep * n_steps  # Total number of timesteps for training
    check_freq = n_steps // 4  # Frequency of checking for best model to save

    # Initialize the environment
    env = FlowControlEnv()
    
    # Path to the model
    model_path = os.path.join(log_dir, "ppo_flow_control_trained.zip")

    # Load or initialize the model
    if not restart and os.path.exists(model_path):
        # Load existing model
        model = PPO.load(model_path, env=env)
        logger.info(f"Loaded existing model for further training for process {process_id}\n")
    else:
        # Train from scratch
        model = PPO('MlpPolicy', env, verbose=0, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, ent_coef=0.01, gamma=gamma)
        if restart:
            logger.info(f"Restarting training from scratch for process {process_id}\n")
        else:
            logger.info(f"No existing model found, starting training from scratch for process {process_id}\n")

    # Print the model architecture
    print_model_architecture(model)

    # Create the callback for saving the best model and logging progress
    save_callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir, total_timesteps=total_timesteps)

    try:
        # Train the agent
        logger.info(f"Starting model training for process {process_id}\n")
        model.learn(total_timesteps=total_timesteps, callback=[save_callback])

        # Save the model
        model.save(model_path)
        logger.info(f"Model saved successfully for process {process_id}\n")

        # Save the environment state using joblib
        saveable_state = env.get_saveable_state()
        dump(saveable_state, os.path.join(log_dir, "flow_control_env_state.joblib"))
        logger.info(f"Environment state saved successfully for process {process_id}\n")
        
    except Exception as e:
        logger.error(f"An error occurred in process {process_id}: {e}")

if __name__ == "__main__":
    restart = False  # Set this to True to restart training from scratch
    
    # Define the 10 different configurations to try
    configurations = [
        {"learning_rate": 0.001, "batch_size": 32, "gamma": 0.95, "process_id": 1},
        {"learning_rate": 0.001, "batch_size": 64, "gamma": 0.95, "process_id": 2},
        {"learning_rate": 0.001, "batch_size": 32, "gamma": 0.99, "process_id": 3},
        {"learning_rate": 0.001, "batch_size": 64, "gamma": 0.99, "process_id": 4},
        {"learning_rate": 0.0001, "batch_size": 32, "gamma": 0.95, "process_id": 5},
        {"learning_rate": 0.0001, "batch_size": 64, "gamma": 0.95, "process_id": 6},
        {"learning_rate": 0.0001, "batch_size": 32, "gamma": 0.99, "process_id": 7},
        {"learning_rate": 0.0001, "batch_size": 64, "gamma": 0.99, "process_id": 8},
    ]

    # Create a process for each configuration
    processes = []
    for config in configurations:
        p = multiprocessing.Process(target=train_environment, args=(config, restart))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()
