import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data_utilities import PersonActionDataset
from models.baseline_3.model import PersonLevelClassifier
from models.train_utils import train_and_validate, print_model_summary
from loader_utils import load_config ,setup_environment
from loader_utils.helper import set_seed,setup_logger

def get_transforms():
    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
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
    # Parse terminal arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, required=False, help="number of epochs to train")
    args = parser.parse_args()

    # Setup Environment (Auto-detect Kaggle vs Local)
    env = setup_environment(baseline_name="baseline_3")
    config = load_config(args.config)
    set_seed(42)

    logger = setup_logger(env['run_dir'])
    logger.info("Starting Baseline 3 Phase_A Training Pipeline")

    # Construct dynamic paths
    videos_path = os.path.join(env['dataset_root'], config.data['videos_dir'])
    annot_path = os.path.join(env['annot_dir'], config.data['annot_file'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Prepare Data
    train_transform, val_transform = get_transforms()

    train_set = PersonActionDataset(videos_path, annot_path, config.data['video_splits']['train'],
                                    transform=train_transform)
    val_set = PersonActionDataset(videos_path, annot_path, config.data['video_splits']['validation'],
                                  transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=config.training['batch_size'], shuffle=True,
                              num_workers=env['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.training['batch_size'], shuffle=False,
                            num_workers=env['num_workers'], pin_memory=True)

    # Initialize Model & Optimizer
    model = PersonLevelClassifier(num_classes=config.model['num_classes']).to(device)
    print_model_summary(model)

    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=config.training['learning_rate'])

    logger.info("Initializing training loop")

    # Training
    best_model = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        # num_epochs=config.training['epochs'],
        num_epochs=args.epochs,
        device=device,
        run_dir=env['run_dir'],
        save_name=config.model['save_name'],
        logger=logger,
        log_interval=10,
        class_names=config.model.get('num_classes_label', None)
    )
    logger.info("Training Pipeline Complete")