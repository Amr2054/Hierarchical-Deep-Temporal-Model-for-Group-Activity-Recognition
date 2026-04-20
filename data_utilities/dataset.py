import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

person_activity_classes = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
person_activity_labels = {class_name.lower():i for i, class_name in enumerate(person_activity_classes)}


group_activity_classes = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {class_name:i for i, class_name in enumerate(group_activity_classes)}


class PersonActionDataset(Dataset):
    """
    Dataset for Baseline 3: Returns individual cropped players and their action (9 classes)
    """

    def __init__(self, videos_root, annot_path, vid_indices, transform=None):
        self.videos_root = videos_root
        self.transform = transform
        self.samples = []
        with open(annot_path, 'rb') as file:
            videos_annot = pickle.load(file)

        for vid_idx in vid_indices:
            vid_idx = str(vid_idx)
            if vid_idx not in videos_annot: continue

            clips = videos_annot[vid_idx]

            for clip_dir, clip_data in clips.items():
                frame_boxes_dct = clip_data['frame_boxes_dct']

                for frame, boxes_info in frame_boxes_dct.items():
                    image_path = os.path.join(self.videos_root, vid_idx, clip_dir, f'{frame}.jpg')
                    if os.path.exists(image_path):
                        for box_info in boxes_info:
                            player_label = person_activity_labels[box_info.category.lower()]
                            self.samples.append((image_path, box_info.box, player_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box, label = self.samples[idx]


        image = cv2.imread(img_path)
        if image is None:
            # Fallback for corrupted/missing images to prevent crashes
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Unpack the box
        xmin, xmax, ymin, ymax = box

        if xmax <= xmin: xmax = xmin + 1
        if ymax <= ymin: ymax = ymin + 1

        # Crop Player Image using box
        cropped_image = image[ymin:ymax, xmin:xmax]

        if self.transform:
            augmented = self.transform(image=cropped_image)
            cropped_image = augmented['image']

        return cropped_image, label


class GroupActivityDataset(Dataset):
    def __init__(self, videos_root, annot_path, vid_indices, transform=None, baseline=1, max_players=12):
        """
        Return frame, and it's label and optionally the bounding boxes info for each player in the frame (for baseline 3)
        """
        self.videos_root = videos_root
        self.transform = transform
        self.baseline = baseline
        self.max_players = max_players
        self.samples = []

        with open(annot_path, 'rb') as file:

            videos_annot = pickle.load(file)

        for vid_idx in vid_indices:
            vid_idx = str(vid_idx)
            if vid_idx not in videos_annot: continue

            clips = videos_annot[vid_idx]

            for clip_dir, clip_data in clips.items():
                clip_label = group_activity_labels[clip_data['category']]

                frame_boxes_dct = clip_data['frame_boxes_dct']

                for frame_id, boxes_info in frame_boxes_dct.items():
                    image_path = os.path.join(self.videos_root, vid_idx, clip_dir, f'{frame_id}.jpg')

                    if os.path.exists(image_path):
                        self.samples.append((image_path, clip_label, boxes_info))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, boxes_info = self.samples[idx]

        image = cv2.imread(image_path)
        if image is None:
            # Fallback for corrupted images
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.baseline == 1:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, label

        elif self.baseline == 3:
            player_crops = []

            for box_info in boxes_info:
                xmin, xmax, ymin, ymax = box_info.box

                if xmax <= xmin: xmax = xmin + 1
                if ymax <= ymin: ymax = ymin + 1

                cropped_image = image[ymin:ymax, xmin:xmax]

                if self.transform:
                    augmented = self.transform(image=cropped_image)
                    cropped_image = augmented['image']
                player_crops.append(cropped_image)

            # Pad with zeros if there are less than max_players detected in the frame
            while len(player_crops) < self.max_players:
                padding = player_crops[0].shape if player_crops else (3, 224, 224)
                player_crops.append(torch.zeros(padding))

            player_crops = player_crops[:self.max_players]
            players_tensor = torch.stack(player_crops)

            return players_tensor, label
