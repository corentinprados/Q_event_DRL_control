import logging
import os

def setup_logging(log_dir, log_filename="test.log"):
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        filename=log_filepath,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    
    # Log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s: %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    # Diagnostic log message to check file writing
    if os.path.exists(log_filepath):
        logger.info(f"Log file {log_filepath} created successfully")
    else:
        logger.error(f"Failed to create log file {log_filepath}")

if __name__ == "__main__":
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)
    logger.info("This is a test log message")
