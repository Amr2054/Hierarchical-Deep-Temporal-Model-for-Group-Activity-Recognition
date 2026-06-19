import os
import torch.multiprocessing as mp
from datetime import datetime
from .config import get_settings

def is_kaggle():
    """Returns True if the code is running on a Kaggle Notebook."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''


def setup_environment(baseline_name="baseline_1"):
    """
    Sets up paths. Dynamically routes outputs to the specific baseline folder.
    Creates a unique timestamped folder for each run to prevent overwriting.
    """
    settings = get_settings()
    env_config = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    parts = baseline_name.split('_')
    if len(parts) >= 2 and parts[0].lower() == 'baseline':
        core_folder = f"{parts[0].lower()}_{parts[1]}"  # Normalizes to "baseline_x"
    else:
        core_folder = baseline_name

    if is_kaggle():
        print("Detected Kaggle Environment.")
        mp.set_start_method('spawn', force=True)
        env_config['dataset_root'] = settings.KAGGLE_DATASET_ROOT

        # Route to specific baseline folder in Kaggle working dir
        base_output = f"/kaggle/working/Group-Activity-Recognition/models/{core_folder}/outputs"
        env_config['num_workers'] = settings.KAGGLE_NUM_WORKERS
        env_config['annot_dir'] = settings.KAGGLE_ANNOT_DIR

        env_config['b3_phase_A_model'] = settings.KAGGLE_B3_PHASE_A_MODEL
        env_config['b5_phase_A_model'] = settings.KAGGLE_B5_PHASE_A_MODEL

    else:
        print("Detected Local Environment.")
        env_config['dataset_root'] = settings.LOCAL_DATASET_ROOT

        # Route to specific baseline folder locally
        base_output = f"./models/{core_folder}/outputs"
        env_config['num_workers'] = settings.LOCAL_NUM_WORKERS
        env_config['annot_dir'] = env_config['dataset_root']

        env_config['b3_phase_A_model'] = settings.LOCAL_B3_PHASE_A_MODEL
        env_config['b5_phase_A_model'] = settings.LOCAL_B5_PHASE_A_MODEL

    # Create a unique run directory
    env_config['run_dir'] = os.path.join(base_output, f"{baseline_name}_run_{timestamp}")
    os.makedirs(env_config['run_dir'], exist_ok=True)

    return env_config