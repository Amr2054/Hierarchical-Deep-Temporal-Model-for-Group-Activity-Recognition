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

class SequenceActivityDataset(Dataset):
    def __init__(self, videos_root, annot_path, vid_indices, transform=None, seq_length =9):
        self.videos_root = videos_root
        self.transform = transform
        self.seq_length = seq_length
        self.samples = []


        with open(annot_path, 'rb') as file:

            videos_annot = pickle.load(file)

        # Iterate on each video
        for vid_idx in vid_indices:
            vid_idx = str(vid_idx)
            if vid_idx not in videos_annot: continue

            clips = videos_annot[vid_idx]
            # Iterate on each clip
            for clip_dir, clip_data in clips.items():
                # Retrieve clip data (label,frames and boxes)
                clip_label = group_activity_labels[clip_data['category']]
                frame_boxes_dct = clip_data['frame_boxes_dct']
                sorted_frame_ids = sorted(frame_boxes_dct.keys(), key=int)

                frame_paths = []
                for frame_id in sorted_frame_ids:
                    image_path = os.path.join(self.videos_root, vid_idx, clip_dir, f'{frame_id}.jpg')

                    if os.path.exists(image_path):
                        frame_paths.append(image_path)

                # 9 frames or pics paths for each clip
                if len(frame_paths) >= self.seq_length:
                    self.samples.append((frame_paths[:self.seq_length],clip_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        frame_paths, label= self.samples[idx]

        clip_frames = []
        for frame_path in frame_paths:
            image = cv2.imread(frame_path)
            if image is None:
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                augmented = self.transform(image=image)
                image_tensor = augmented['image']
            else:
                # Failsafe if no transform is provided
                image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

            clip_frames.append(image_tensor)

        # Shape : (9,3,244,244)
        # dim = 0 : means stack these tensors in dimension index 0 of the shape
        sequence_tensor = torch.stack(clip_frames,dim=0)  # before : 9 separate tensors , after : 4D tensor (frames,c,h,w)
        return sequence_tensor, label

# class PlayerSequenceActivityDataset(Dataset):
#     def __init__(self, videos_root, annot_path, vid_indices, transform=None, seq_length=9, max_players=12):
#         self.videos_root = videos_root
#         self.transform = transform
#         self.seq_length = seq_length
#         self.max_players = max_players
#         self.samples = []
#
#         with open(annot_path, 'rb') as file:
#             videos_annot = pickle.load(file)
#
#         # Iterate on each video
#         for vid_idx in vid_indices:
#             vid_idx = str(vid_idx)
#             if vid_idx not in videos_annot: continue
#
#             clips = videos_annot[vid_idx]
#             # Iterate on each clip
#             for clip_dir, clip_data in clips.items():
#                 # Retrieve clip data (label,frames and boxes)
#                 clip_label = group_activity_labels[clip_data['category']]
#                 frame_boxes_dct = clip_data['frame_boxes_dct']
#                 sorted_frame_ids = sorted(frame_boxes_dct.keys(), key=int)
#
#                 frame_data_list = []
#                 for frame_id in sorted_frame_ids:
#                     image_path = os.path.join(self.videos_root, vid_idx, clip_dir, f'{frame_id}.jpg')
#                     if os.path.exists(image_path):
#                         frame_data_list.append({
#                             'path': image_path,
#                             'boxes': frame_boxes_dct[frame_id], # for each frame we have 12 boxes
#                         })
#                 self.samples.append((frame_data_list,clip_label))
#
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#
#         frame_data_list, label = self.samples[idx]
#
#         sequence_crops = []
#
#         for frame_data in frame_data_list:
#             image_path = frame_data['path']
#             boxes_info = frame_data['boxes']
#
#             image = cv2.imread(image_path)
#             if image is None:
#                 image = np.zeros((720, 1280, 3), dtype=np.uint8)
#             else:
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#             players = []
#             for box_info in boxes_info:
#                 # Unpack the box
#                 xmin, xmax, ymin, ymax = box_info.box
#
#                 if xmax <= xmin: xmax = xmin + 1
#                 if ymax <= ymin: ymax = ymin + 1
#
#                 # Crop Player Image using box
#                 cropped_image= image[ymin:ymax, xmin:xmax]
#
#                 if self.transform:
#                     augmented = self.transform(image=cropped_image)
#                     cropped_tensor = augmented['image']
#                 else:
#                     cropped_tensor = torch.from_numpy(cropped_image.transpose(2, 0, 1)).float()
#                 players.append(cropped_tensor) # players at a specific frame
#
#             # Handle varying number of players (Pad with zeros or truncate to exactly 12)
#             while len(players) < self.max_players:
#                 padding = players[0].shape if players else (3, 224, 224)
#                 players.append(torch.zeros(padding))
#
#             players = players[:self.max_players]
#
#             # Stack the 12 players into a single tensor for this frame -> [12, 3, 224, 224]
#             frame_tensor = torch.stack(players, dim=0)
#             sequence_crops.append(frame_tensor)
#
#         # Stack the 9 frames together -> [9, 12, 3, 224, 224] (This is [Time, Players, C, H, W])
#         clip_tensor = torch.stack(sequence_crops, dim=0)
#
#         # The Dimension Flip
#         # The LSTM needs tracking per player, so it wants the Player dimension first.
#         # .permute() swaps the axes: dim 1 (Players) moves to position 0, dim 0 (Time) moves to position 1
#         final_tensor = clip_tensor.permute(1, 0, 2, 3, 4)
#
#         return final_tensor, label


class PlayerSequenceActivityDataset(Dataset):
    """
    Dataset for Phase A:
    Extracts a temporal sequence (9 frames) of a SINGLE player.
    Assigns the label from the Anchor Keyframe (the middle frame).
    Output: [Seq_Len, Channels, Height, Width], Label
    """

    def __init__(self, videos_root, annot_path, vid_indices, transform=None, seq_length=9):
        self.videos_root = videos_root
        self.transform = transform
        self.seq_length = seq_length
        self.samples = []

        with open(annot_path, 'rb') as file:
            videos_annot = pickle.load(file)

        for vid_idx in vid_indices:
            vid_idx = str(vid_idx)
            if vid_idx not in videos_annot: continue

            clips = videos_annot[vid_idx]

            for clip_dir, clip_data in clips.items():
                frame_boxes_dct = clip_data['frame_boxes_dct']
                sorted_frame_ids = sorted(frame_boxes_dct.keys(), key=int)

                # Group boxes by player_ID to track individuals over time
                player_tracks = {i: [] for i in range(12)}

                for frame_id in sorted_frame_ids:
                    image_path = os.path.join(self.videos_root, vid_idx, clip_dir, f'{frame_id}.jpg')
                    if not os.path.exists(image_path):
                        continue

                    for box_info in frame_boxes_dct[frame_id]:
                        if box_info.player_ID < 12:
                            player_tracks[box_info.player_ID].append({
                                'path': image_path,
                                'box': box_info.box,
                                'category': box_info.category
                            })

                # Validate and save sequences that have the full 9 frames
                for player_id, track in player_tracks.items():
                    if len(track) >= self.seq_length:

                        # Extract the Anchor Keyframe label (the middle frame)
                        center_idx = self.seq_length // 2  # For 9 frames, this is index 4
                        label_str = track[center_idx]['category'].lower()

                        person_label = person_activity_labels[label_str]

                        self.samples.append((track[:self.seq_length], person_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        track_data, label = self.samples[idx]

        sequence_crops = []

        for frame_data in track_data:
            image = cv2.imread(frame_data['path'])
            if image is None:
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            xmin, xmax, ymin, ymax = frame_data['box']
            if xmax <= xmin: xmax = xmin + 1
            if ymax <= ymin: ymax = ymin + 1

            cropped_image = image[ymin:ymax, xmin:xmax]

            if self.transform:
                augmented = self.transform(image=cropped_image)
                cropped_tensor = augmented['image']
            else:
                cropped_tensor = torch.from_numpy(cropped_image.transpose(2, 0, 1)).float()

            sequence_crops.append(cropped_tensor)

        # Stack the 9 frames together -> [9, 3, 224, 224]
        sequence_tensor = torch.stack(sequence_crops, dim=0)

        return sequence_tensor, label