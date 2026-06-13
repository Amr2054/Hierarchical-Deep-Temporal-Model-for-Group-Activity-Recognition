import argparse
import os

import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from data import GroupActivityDataset
from models import train_and_validate, print_model_summary
from models.baseline_3.model import PersonLevelClassifier, GroupLevelClassifier
from utils import load_config, set_seed, setup_logger, setup_environment


def get_transforms():
    """
    Standard spatial geometry for the ResNet50 backbone.
    """
    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.ColorJitter(brightness=0.1, contrast=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, required=False, help="number of epochs to train")
    args = parser.parse_args()

    # Environment setup
    env = setup_environment(baseline_name="baseline_3_phase_B")
    config = load_config(args.config)
    set_seed(42)

    # Initialize Logger
    logger = setup_logger(env['run_dir'])
    logger.info("Starting Baseline 3 Phase_B Training Pipeline...")

    # Dynamic Paths based on environment type
    videos_path = os.path.join(env['dataset_root'], config.data['videos_dir'])
    annot_path = os.path.join(env['annot_dir'], config.data['annot_file'])
    person_weights_path = env['b3_phase_A_model']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_transform, val_transform = get_transforms()

    train_set = GroupActivityDataset(videos_path, annot_path, config.data['video_splits']['train'],
                                     transform=train_transform, baseline=3)
    val_set = GroupActivityDataset(videos_path, annot_path, config.data['video_splits']['validation'],
                                     transform=val_transform, baseline=3)

    train_loader = DataLoader(train_set, batch_size=config.training['batch_size'], shuffle=True,
                              num_workers=env['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.training['batch_size'], shuffle=False,
                              num_workers=env['num_workers'], pin_memory=True)

    # Load Phase A Model & Weights
    person_model = PersonLevelClassifier(num_classes=config.model['num_person_classes'])
    try:
        checkpoint = torch.load(person_weights_path, map_location=device)
        person_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded Phase A weights from: {person_weights_path}")

    except FileNotFoundError:
        print(f"Error: {person_weights_path} not found. You must train Phase A first")
        exit(1)

    # Initialize Model, Loss, Optimizer
    model = GroupLevelClassifier(person_classifier=person_model, num_classes=config.model['num_group_classes']).to(device)
    print_model_summary(model)

    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=config.training['learning_rate'],
                            weight_decay=config.training['weight_decay'])

    logger.info("Initializing training loop")
    num_epochs = args.epochs if args.epochs else config.training['epochs']

    # Training Model
    best_model = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        run_dir= env['run_dir'],
        save_name= config.model['save_name'],
        logger=logger,
        class_names=config.model.get('group_activity', None),
        early_stop_patience=config.training['early_stop_patience']
    )
    logger.info("Baseline 3 Phase B Training Complete!")