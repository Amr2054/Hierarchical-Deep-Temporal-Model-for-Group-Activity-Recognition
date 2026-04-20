import yaml
import random
import numpy as np
import torch
import logging
import os

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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Also prints to terminal
        ]
    )
    return logging.getLogger()