import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import PlayerGroupActivityDataset
from models.baseline_5.model import Person_Activity_Temporal_Classifier
from models.baseline_7.model import  Full_Hierarchical_Model
from models import train_and_validate, print_model_summary
from utils import load_config, set_seed, setup_logger,setup_environment


def get_transforms():
    """
    Locked spatial geometry for the LSTM sequences.
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
    parser.add_argument("--epochs", type=int, required=False, help="Number of epochs to train")
    args = parser.parse_args()

    # Environment Setup
    env = setup_environment(baseline_name="baseline_7")
    config = load_config(args.config)
    set_seed(42)

    # Initialize Logger
    logger = setup_logger(env['run_dir'])
    logger.info("Starting Baseline 7 (Full Hierarchical Model) Training Pipeline")

    # Dynamic Paths based on environment type
    videos_path = os.path.join(env['dataset_root'], config.data['videos_dir'])
    annot_path = os.path.join(env['annot_dir'], config.data['annot_file'])
    person_weights_path = env['b5_phase_A_model']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Data Preparation
    train_transform, val_transform = get_transforms()

    train_set = PlayerGroupActivityDataset(
        videos_root=videos_path,
        annot_path=annot_path,
        vid_indices=config.data['video_splits']['train'],
        transform=train_transform,
        seq_length=9,
        max_players=12
    )

    val_set = PlayerGroupActivityDataset(
        videos_root=videos_path,
        annot_path=annot_path,
        vid_indices=config.data['video_splits']['validation'],
        transform=val_transform,
        seq_length=9,
        max_players=12
    )

    train_loader = DataLoader(train_set, batch_size=config.training['batch_size'], shuffle=True,
                              num_workers=env['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.training['batch_size'], shuffle=False,
                              num_workers=env['num_workers'], pin_memory=True)

    # Load B5 Phase A Model (LSTM 1)
    person_model = Person_Activity_Temporal_Classifier(
        input_size=config.model['input_size'],
        num_classes=config.model['num_person_classes'],
        hidden_size=config.model['hidden_person_size'],
        num_layers=config.model['num_person_layers']
    )
    try:
        checkpoint = torch.load(person_weights_path, map_location=device)
        person_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded Phase A weights from: {person_weights_path}")

    except FileNotFoundError:
        print(f"Error: {person_weights_path} not found. You must train Phase A first")
        exit(1)

    # Initialize Baseline 7 Model
    model = Full_Hierarchical_Model(
        person_classifier=person_model,
        num_classes=config.model['num_classes'],
        input_size=config.model['input_size'],
        hidden1_size=config.model['hidden1_size'],
        hidden2_size=config.model['hidden2_size'],
        num_layers=config.model['num_layers']
    ).to(device)
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
        run_dir=env['run_dir'],
        save_name=config.model['save_name'],
        logger=logger,
        class_names=config.model.get('num_classes_label', None),
        early_stop_patience=config.training['early_stop_patience'],
    )

    logger.info("Baseline 7 Training Complete!")