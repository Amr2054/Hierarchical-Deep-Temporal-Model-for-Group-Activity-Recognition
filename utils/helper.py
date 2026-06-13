import yaml
import random
import numpy as np
import torch
import logging
import os
import sys

class Config:
    def __init__(self, config_dict):
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})
        self.data = config_dict.get("data", {})

    def __repr__(self):
        return f"Config(model={self.model}, training={self.training}, data={self.data})"

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = Config(config)
    return config

def set_seed(seed=42):
    """Locks all random operations for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def setup_logger(run_dir):
    """Creates a text file that records every print statement."""
    log_file = os.path.join(run_dir, 'training.log')


    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if the function is called twice
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler (saves to the .log file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (prints to the terminal)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def calculate_balanced_weights(train_set,min_length):

    all_train_labels = [sample[1] for sample in train_set.samples]
    class_counts = np.bincount(all_train_labels, minlength=min_length)
    total_samples = len(all_train_labels)
    num_classes_active = len([c for c in class_counts if c > 0])
    class_weights = total_samples / (num_classes_active * (class_counts + 1e-6))

    return class_weights
