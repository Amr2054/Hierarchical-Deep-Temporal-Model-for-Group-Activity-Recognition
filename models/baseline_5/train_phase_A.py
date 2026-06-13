import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import PlayerSequenceActivityDataset
from models.baseline_5.model import Person_Activity_Temporal_Classifier
from models import train_and_validate, print_model_summary
from utils import load_config, set_seed, setup_logger, setup_environment, calculate_balanced_weights

def get_transforms():
    """
    Locked spatial geometry for LSTM sequences.
    """
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
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
    env = setup_environment(baseline_name="baseline_5")
    config = load_config(args.config)
    set_seed(42)

    # Initialize Logger
    logger = setup_logger(env['run_dir'])
    logger.info("Starting Baseline 5 Phase A Training Pipeline")

    # Dynamic Paths based on environment type
    videos_path = os.path.join(env['dataset_root'], config.data['videos_dir'])
    annot_path = os.path.join(env['annot_dir'], config.data['annot_file'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Data Preparation
    train_transform, val_transform = get_transforms()

    train_set = PlayerSequenceActivityDataset(videos_path, annot_path, config.data['video_splits']['train'],
                                              transform=train_transform, seq_length=9)
    val_set = PlayerSequenceActivityDataset(videos_path, annot_path, config.data['video_splits']['validation'],
                                              transform=val_transform, seq_length=9)

    train_loader = DataLoader(train_set, batch_size=config.training['batch_size'], shuffle=True,
                              num_workers=env['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.training['batch_size'], shuffle=False,
                              num_workers=env['num_workers'], pin_memory=True)

    # Initialize Model, Loss, Optimizer
    model = Person_Activity_Temporal_Classifier(
        input_size=config.model['input_size'],
        num_classes=config.model['num_classes'],
        hidden_size=config.model['hidden_size'],
        num_layers=config.model['num_layers']
    ).to(device)
    print_model_summary(model)

    # Dynamic Class Weighting
    logger.info("Calculating dynamic class weights from the dataset")
    class_weights = calculate_balanced_weights(train_set=train_set,min_length=config.model['num_classes'])
    weights_tensor = torch.FloatTensor(class_weights).to(device)

    logger.info(f"Applied Class Weights: {np.round(class_weights, 3)}")
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Optimizer
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
        run_dir=env['run_dir'],
        save_name=config.model['save_name'],
        logger=logger,
        class_names=config.model.get('num_classes_label', None),
        early_stop_patience=config.model['early_stop_patience'],
    )

    logger.info("Baseline 5 Phase A Training Complete")