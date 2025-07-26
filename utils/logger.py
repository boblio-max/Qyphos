import logging
import sys

def setup_logger():
    """Sets up a colored, formatted logger."""
    logger = logging.getLogger("Qyphos")
    if logger.hasHandlers():
        return logger # Avoid adding duplicate handlers

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    
    # Simple formatter for now, can be expanded with colors
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

log = setup_logger()
