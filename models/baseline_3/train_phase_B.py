import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data_utilities import GroupActivityDataset
from models.baseline_3.model import PersonLevelClassifier, GroupLevelClassifier
from models.train_utils import train_and_validate
from loader_utils.helper import load_config
from loader_utils.env_utils import setup_environment
from loader_utils.helper import set_seed,setup_logger
from models import print_model_summary

def get_transforms():
    base_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return base_transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Environment setup
    env = setup_environment(baseline_name="baseline_3")
    config = load_config(args.config)
    set_seed(42)

    logger = setup_logger(env['run_dir'])
    logger.info("Starting Baseline 3 Phase_B Training Pipeline...")

    # Dynamic Paths
    videos_path = os.path.join(env['dataset_root'], config.data['videos_dir'])
    annot_path = os.path.join(env['annot_dir'], config.data['annot_file'])

    person_weights_path = os.path.join(env['run_dir'], config.model['person_weights_name'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Group Classifier (Phase B) on device: {device}")

    base_transform = get_transforms()

    train_set = GroupActivityDataset(videos_path, annot_path, config.data['video_splits']['train'], transform=base_transform, baseline=3)
    val_set = GroupActivityDataset(videos_path, annot_path, config.data['video_splits']['validation'], transform=base_transform, baseline=3)

    train_loader = DataLoader(train_set, batch_size=config.training['batch_size'], shuffle=True, num_workers=env['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.training['batch_size'], shuffle=False, num_workers=env['num_workers'], pin_memory=True)

    # Load Phase A Model & Weights
    person_model = PersonLevelClassifier(num_classes=config.model['num_person_classes'])
    try:
        person_model.load_state_dict(torch.load(person_weights_path, map_location=device, weights_only=True))
        print(f"Successfully loaded Phase A weights from: {person_weights_path}")
    except FileNotFoundError:
        print(f"Error: {person_weights_path} not found. You must train Phase A first")
        exit(1)

    # Initialize Phase B Model
    model = GroupLevelClassifier(person_classifier=person_model, num_classes=config.model['num_group_classes']).to(device)
    print_model_summary(model)

    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=config.training['learning_rate'])

    logger.info("Initializing training loop")

    best_model = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.training['epochs'],
        device=device,
        run_dir= env['run_dir'],
        save_name= config.model['save_name'],
        logger=logger,
        log_interval=10,
        class_names=config.model.get('group_activity', None)
    )
    logger.info("Training Pipeline Complete!")