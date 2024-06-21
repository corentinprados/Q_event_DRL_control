#log_analyzer.py

import re

def analyze_log_file(filename):
    """
    Analyzes the log file to count the average number of actions per episode,
    the average number of steps per episode, and the average duration of episodes.

    Args:
        filename (str): The path to the log file to analyze.

    Returns:
        tuple: A tuple containing the average number of actions per episode,
               the average number of steps per episode, and the average duration of episodes.
    """
    # Lists to store the number of actions, steps, and durations per episode
    episode_actions = []
    episode_steps = []
    episode_durations = []
    current_actions = 0  # Counter for actions in the current episode
    current_steps = 0    # Counter for steps in the current episode
    current_duration = 0.0  # Duration of the current episode
    episode_started = True  # Flag to indicate if an episode has started

    # Define regular expressions to extract episode information
    episode_regex = re.compile(r'Episode (\d+) finished after (\d+) steps')
    duration_regex = re.compile(r'Episode \d+ duration: ([\d\.]+) seconds')

    # Open and read the log file
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Check if the line indicates the end of an episode
            if 'Maximum simulation time reached' in line:
                if episode_started:
                    episode_actions.append(current_actions)  # Add the actions of the episode to the list
                    current_actions = 0  # Reset the actions counter for the next episode
                    episode_started = False  # Indicate that the episode has ended
            else:
                # Count the occurrences of the word 'Action' in the line and add them to the current counter
                action_count = line.count('Action')
                if action_count > 0:
                    current_actions += action_count
                
                # Check if the line contains episode information
                episode_match = episode_regex.search(line)
                if episode_match:
                    episode_num = int(episode_match.group(1))  # Extract the episode number
                    steps = int(episode_match.group(2))  # Extract the number of steps
                    episode_steps.append(steps)  # Add the number of steps to the list
                    episode_started = True  # Indicate that a new episode has started
                
                # Check if the line contains duration information
                duration_match = duration_regex.search(line)
                if duration_match:
                    duration = float(duration_match.group(1))  # Extract the duration
                    episode_durations.append(duration)  # Add the duration to the list

    # Append the last episode if the file doesn't end with 'Maximum simulation time reached'
    if current_actions > 0:
        episode_actions.append(current_actions)

    # Calculate the averages and round to the nearest integer
    avg_actions = round(sum(episode_actions) / len(episode_actions)) if episode_actions else 0
    avg_steps = round(sum(episode_steps) / len(episode_steps)) if episode_steps else 0
    avg_duration = round(sum(episode_durations) / len(episode_durations), 2) if episode_durations else 0.0

    return avg_actions, avg_steps, avg_duration

# Script usage
if __name__ == "__main__":
    for i in range(1,9):
        filename = f"../tmp/process_{i}/training.log"  # Path to the log file
        avg_actions, avg_steps, avg_duration = analyze_log_file(filename)  # Analyze the log file
        avg_duration_minutes = round(avg_duration / 60, 2)  # Convert average duration to minutes
        print(f"The average number of actions per episode is {avg_actions}")  # Print the average number of actions
        print(f"The average number of steps per episode is {avg_steps}")  # Print the average number of steps
        print(f"The average duration per episode is {avg_duration} seconds ({avg_duration_minutes} minutes)")  # Print the average duration in seconds and minutes
