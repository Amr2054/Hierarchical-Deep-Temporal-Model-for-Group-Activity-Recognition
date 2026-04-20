import os
import torch.multiprocessing as mp
from datetime import datetime

def is_kaggle():
    """Returns True if the code is running on a Kaggle Notebook."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''


def setup_environment(baseline_name="baseline_1"):
    """
    Sets up paths. Dynamically routes outputs to the specific baseline folder.
    Creates a unique timestamped folder for each run to prevent overwriting.
    """
    env_config = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if is_kaggle():
        print("Detected Kaggle Environment.")
        mp.set_start_method('spawn', force=True)
        env_config['dataset_root'] = "/kaggle/input/datasets/ahmedmohamed365/volleyball"
        # Route to specific baseline folder in Kaggle working dir
        base_output = f"/kaggle/working/models/{baseline_name}/outputs"
        env_config['num_workers'] = 4
        env_config['annot_dir'] = "/kaggle/working"
    else:
        print("Detected Local Environment.")
        env_config[
            'dataset_root'] = "/home/amr/Study/Volley Ball Project/A Hierarchical Deep Temporal Model for Group Activity Recognition/dataset"
        # Route to specific baseline folder locally
        base_output = f"./models/{baseline_name}/outputs"
        env_config['num_workers'] = 4
        env_config['annot_dir'] = env_config['dataset_root']
    # Create a unique run directory
    env_config['run_dir'] = os.path.join(base_output, f"run_{timestamp}")
    os.makedirs(env_config['run_dir'], exist_ok=True)

    return env_config