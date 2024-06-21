# Flow Control with Reinforcement Learning

This project demonstrates the use of Reinforcement Learning (RL) to control flow dynamics using the `Dedalus` library for solving partial differential equations. The RL environment is implemented using the `gymnasium` library and the PPO algorithm from `stable-baselines3`.

## Project Structure

Thera two folder names: 
- `my_project_without_wandb`: my_project only on local.
- `my_project_with_wandb`: my_project using wandb, a service to stock online machine learning informatioins
There are two main versions of the project:
- `my_project_without_wandb`: This version of the project runs entirely on the local machine without using any external services.
- `my_project_with_wandb`: This version of the project integrates with Weights & Biases (wandb), a service for tracking and managing machine learning experiments online.


```
my_project/
├── flow_control/
│   ├── __init__.py
│   ├── flow_control_env.py
│   ├── callbacks.py
├── main_program.py
├── multi_training.py
├── logging_setup.py
├── post_analysis/
│   ├── log_analyzer.py
│   ├── test_model.py
│   ├── plot_training_rewards.py
│   ├── plot_episode_rewards.py
│   └── checkpoints_initial_conditions/
│       ├── checkpoints_s24/
│       └── ...
└── checkpoints_initial_conditions/
    ├── checkpoints_s24.h5
    └── checkpoints_s24/
        ├── checkpoints_s24_p0.h5
        ├── ...
```

### Directory and File Descriptions

#### `flow_control/`

This directory contains the core components for the RL environment and callback functionality.

- `__init__.py`: Marks the directory as a Python package. It can be empty or contain package initialization code.
- `flow_control_env.py`: Contains the `FlowControlEnv` class, which defines the RL environment for controlling the flow dynamics using Dedalus simulations.
- `callbacks.py`: Contains the `SaveOnBestTrainingRewardCallback` class, which is used to save the best model during training based on the reward received.

#### `main_program.py`

This script is the main entry point for training the RL agent. It sets up the environment, defines the PPO model, and includes functionality for quick training of the agent. It also uses a callback to save the best model during training.

#### `multi_training.py`

This script allows for the training of multiple RL agents in parallel.

#### `logging_setup.py`

This script sets up the logging configuration for the project, ensuring consistent and comprehensive logging across all components.

#### `post_analysis/`

This directory contains scripts for post-training analysis.

- `log_analyzer.py`: Analyzes the logs generated during training to provide insights and statistics.
- `test_model.py`: Loads the trained model and the saved environment to test the performance of the RL agent.
- `plot_training_rewards.py`: Plots the rewards obtained during training for visualization purposes.
- `plot_episode_rewards.py`: Plots the rewards obtained per episode for further analysis.
- `checkpoints_initial_conditions/`: Contains initial condition checkpoint files used during the simulation.
