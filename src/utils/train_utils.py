
import os
import torch
import torch.nn as nn
import logging
from logging import StreamHandler, FileHandler

def get_logger(log_dir: str, 
               log_filename: str = 'training.log'):
    
    # Create a custom logger
    logger = logging.getLogger('TrainerLogger')
    logger.setLevel(logging.INFO)

    # Create handlers for both console and file output
    c_handler = StreamHandler()
    f_handler = FileHandler(f'{log_dir}/{log_filename}', mode='a')  # Append mode

    # Set level for handlers (info level logs)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatter for a consistent log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to handlers
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def save_model_weights(model:nn.Module, epoch: int, weights_dir:str) -> None:
    """
    Save model weights to a file.
    """
    weights = model.state_dict()
    model_path = os.path.join(weights_dir, f"model_epoch_{epoch}.pth")
    torch.save(weights, model_path)
    print(f"Model weights saved to {model_path}")