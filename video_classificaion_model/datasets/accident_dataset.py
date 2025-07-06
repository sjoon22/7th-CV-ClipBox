import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torch

class AccidentDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, val_ratio=0.1, seed=42):
        """
        Args:
            root_dir (str): /data/datasets/car_accident
            mode (str): 'train' or 'val'
            transform: torchvision transforms
            val_ratio (float): validation split 비율 (e.g., 0.1 = 10%)
            seed (int): 랜덤 시드 고정
        """
        assert mode in ['train', 'val'], "mode should be 'train' or 'val'"
        self.mode = mode
        self.transform = transform
        self.samples = []

        # seed 고정
        random.seed(seed)

        # abnormal (label=1)
        self._split_class(os.path.join(root_dir, 'abnormal'), label=1, val_ratio=val_ratio)

        # normal (label=0)
        self._split_class(os.path.join(root_dir, 'normal'), label=0, val_ratio=val_ratio)

    def _split_class(self, class_dir, label, val_ratio):
        """
        하나의 클래스(abnormal or normal)를 train/val로 분할하여 self.samples에 추가
        """
        all_folders = sorted([
            os.path.join(class_dir, d)
            for d in os.listdir(class_dir)
            if os.path.isdir(os.path.join(class_dir, d))
        ])
        random.shuffle(all_folders)

        n_total = len(all_folders)
        n_val = int(n_total * val_ratio)

        if self.mode == 'val':
            selected = all_folders[:n_val]
        else:  # 'train'
            selected = all_folders[n_val:]

        for folder in selected:
            self.samples.append((folder, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]

        frame_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        frames = []
        for fname in frame_files:
            img_path = os.path.join(folder_path, fname)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        # [T, C, H, W]
        video_tensor = torch.stack(frames, dim=0)
        return {
            'video': video_tensor,
            'label': label,
            'path': folder_path
        }