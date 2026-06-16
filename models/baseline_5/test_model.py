import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import PlayerGroupActivityDataset
from models.baseline_5.model import Person_Activity_Temporal_Classifier,Group_Activity_Classifier

from models import evaluate_test_set
from utils import load_config, set_seed, setup_logger,setup_environment


def get_test_transform():
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline 5 on the Test Set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth checkpoint")
    parser.add_argument("--person_weights", type=str, required=False, help="Path to Phase A weights")
    args = parser.parse_args()

    # Environment Setup
    env = setup_environment(baseline_name="baseline_5_test")
    config = load_config(args.config)
    set_seed(42)

    logger = setup_logger(env['run_dir'],mode="test")
    logger.info(" Starting Baseline 5 Test Evaluation Pipeline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Preparation (Using the TEST split)
    videos_path = os.path.join(env['dataset_root'], config.data['videos_dir'])
    annot_path = os.path.join(env['annot_dir'], config.data['annot_file'])
    test_transform = get_test_transform()

    test_set = PlayerGroupActivityDataset(
        videos_root=videos_path,
        annot_path=annot_path,
        vid_indices=config.data['video_splits']['test'],
        transform=test_transform,
        seq_length=9,
        max_players=12
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.training['batch_size'],
        shuffle=False,
        num_workers=env['num_workers'],
        pin_memory=True
    )

    # Load Phase A Model & Weights
    person_model = Person_Activity_Temporal_Classifier(
        input_size=config.model['input_size'],
        num_classes=config.model['num_person_classes'],
        hidden_size=config.model['hidden_size'],
        num_layers=config.model['num_layers']
    )
    checkpoint = torch.load(args.person_weights, map_location=device)
    person_model.load_state_dict(checkpoint['model_state_dict'])

    model = Group_Activity_Classifier(
        person_feature_extraction=person_model,
        num_classes=config.model['num_classes']
    ).to(device)

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