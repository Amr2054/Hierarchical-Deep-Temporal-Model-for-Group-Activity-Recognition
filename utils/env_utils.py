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

    parts = baseline_name.split('_')
    if len(parts) >= 2 and parts[0].lower() == 'baseline':
        core_folder = f"{parts[0].lower()}_{parts[1]}"  # Normalizes to "baseline_x"
    else:
        core_folder = baseline_name

    if is_kaggle():
        print("Detected Kaggle Environment.")
        mp.set_start_method('spawn', force=True)
        env_config['dataset_root'] = "/kaggle/input/datasets/ahmedmohamed365/volleyball"

        # Route to specific baseline folder in Kaggle working dir
        base_output = f"/kaggle/working/Hierarchical-Deep-Temporal-Model-for-Group-Activity-Recognition/models/{core_folder}/outputs"
        env_config['num_workers'] = 4
        env_config['annot_dir'] = "/kaggle/working"

        # env_config['b3_phase_A_model'] = "/kaggle/input/models/amr2054/best-person-classifier/pytorch/default/1/best_person_classifier.pth"
        env_config['b3_phase_A_model'] = "/kaggle/input/models/amrahmedgohar/best-person-classifier/pytorch/default/1/best_person_classifier.pth"
        env_config['b5_phase_A_model'] = "/kaggle/input/models/amr2054/best-person-temporal-classifier/pytorch/default/1/best_baseline5_phase_A.pth"

    else:
        print("Detected Local Environment.")
        env_config[
            'dataset_root'] = "/home/amr/Study/Volley Ball Project/A Hierarchical Deep Temporal Model for Group Activity Recognition/dataset"

        # Route to specific baseline folder locally
        base_output = f"./models/{core_folder}/outputs"
        env_config['num_workers'] = 4
        env_config['annot_dir'] = env_config['dataset_root']

        env_config['b3_phase_A_model'] = "models/baseline_3/outputs/run_20260515_000626/best_person_classifier.pth"
        env_config['b5_phase_A_model'] = "models/baseline_5/outputs/run_20260608_141138/best_baseline5_phase_A.pth"

    # Create a unique run directory
    env_config['run_dir'] = os.path.join(base_output, f"{baseline_name}_run_{timestamp}")
    os.makedirs(env_config['run_dir'], exist_ok=True)

    return env_config