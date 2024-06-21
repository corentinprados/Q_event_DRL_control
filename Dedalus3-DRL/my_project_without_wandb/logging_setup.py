# logging_setup.py

import logging
import os

def setup_logging(log_dir, log_filename="training.log"):
    """
    Set up logging configuration to log to a file.

    Parameters:
    log_dir (str): Directory where the log file will be saved.
    log_filename (str): Name of the log file. Default is "training.log".
    """
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception as e:
            print(f"Failed to create log directory {log_dir}: {e}")
            return

    # Ensure the log file can be created
    try:
        with open(log_filepath, 'a') as f:
            pass
    except Exception as e:
        print(f"Failed to create log file {log_filepath}: {e}")
        return
    
    # Get the root logger and set the logging level to INFO
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create a console handler to log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)s: %(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log that the logging setup is complete
    logger.info("Logging setup complete")
    
    # Diagnostic log message to check if the log file was created successfully
    if os.path.exists(log_filepath):
        logger.info(f"Log file {log_filepath} created successfully")
    else:
        logger.error(f"Failed to create log file {log_filepath}")

    # Inform the user that the logging setup is complete
    print(f"Log setup complete. Log file: {log_filepath}")

def flush_loggers():
    """
    Flush all log handlers to ensure all logging output has been written.
    """
    for handler in logging.getLogger().handlers:
        handler.flush()
