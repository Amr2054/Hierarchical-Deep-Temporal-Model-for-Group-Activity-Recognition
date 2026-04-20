import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data_utilities import GroupActivityDataset
from models.baseline_1.model import ResNet50FineTuner
from models import train_and_validate
from loader_utils import load_config,set_seed,setup_logger
from loader_utils import setup_environment

def get_transforms():
    train_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.RandomResizedCrop(size=(224, 224)),
        A.Rotate(limit=5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return train_transform, val_transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Environment setup
    env = setup_environment(baseline_name="baseline_1")
    config = load_config(args.config)
    set_seed(42)

    # Initialize Logger
    # logger = setup_logger(env['run_dir'])
    # logger.info("Starting Baseline 1 Training Pipeline...")

    # Dynamic Paths based on environment type
    videos_path = os.path.join(env['dataset_root'], config.data['videos_dir'])
    annot_path = os.path.join(env['dataset_root'], config.data['annot_file'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Baseline 1 on device: {device}")

    # Prepare Data for model
    train_transform, val_transform = get_transforms()

    train_set = GroupActivityDataset(videos_path, annot_path, config.data['video_splits']['train'], transform=train_transform, baseline=1)
    val_set = GroupActivityDataset(videos_path, annot_path, config.data['video_splits']['validation'], transform=val_transform, baseline=1)

    train_loader = DataLoader(train_set, batch_size=config.training['batch_size'], shuffle=True, num_workers=env['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.training['batch_size'], shuffle=False, num_workers=env['num_workers'], pin_memory=True)

    # Initialize Model, Loss, Optimizer
    model = ResNet50FineTuner(num_classes=config.model['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=config.training['learning_rate'])

    # logger.info("Initializing training loop")

    # Train Model
    best_model = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.training['epochs'],
        device=device,
        run_dir=env['run_dir'],
        save_name=config.model['save_name'],
        class_names=config.model.get('num_clases_label', None)
    )
    # logger.info("Training Pipeline Complete")