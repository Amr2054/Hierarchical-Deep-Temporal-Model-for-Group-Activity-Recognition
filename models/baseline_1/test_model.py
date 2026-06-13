import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import GroupActivityDataset
from models.baseline_1.model import ResNet50FineTuner

from models import evaluate_test_set
from utils import load_config, set_seed, setup_logger,setup_environment


def get_test_transform():
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline 1 on the Test Set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth checkpoint")
    parser.add_argument("--person_weights", type=str, required=False, help="Path to Phase A weights")
    args = parser.parse_args()

    # Environment Setup
    env = setup_environment(baseline_name="baseline_1_test")
    config = load_config(args.config)
    set_seed(42)

    logger = setup_logger(env['run_dir'])
    logger.info(" Starting Baseline 1 Test Evaluation Pipeline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Preparation (Using the TEST split)
    videos_path = os.path.join(env['dataset_root'], config.data['videos_dir'])
    annot_path = os.path.join(env['annot_dir'], config.data['annot_file'])
    test_transform = get_test_transform()

    test_set = GroupActivityDataset(
        videos_path,
        annot_path,
        config.data['video_splits']['test'],
        transform=test_transform,
        baseline=1
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.training['batch_size'],
        shuffle=False,
        num_workers=env['num_workers'],
        pin_memory=True
    )

    model = ResNet50FineTuner(num_classes=config.model['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()

    # Execute the Evaluation
    evaluate_test_set(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        checkpoint_path=args.checkpoint,
        run_dir=env['run_dir'],
        logger=logger,
        class_names=config.model.get('num_classes_label', None)
    )