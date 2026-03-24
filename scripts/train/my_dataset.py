import json
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.config import *
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        # Prefer pose-conditioned annotations if available.
        pose_json_path = os.path.join(dataset_root, "train_pose.json")
        basic_json_path = os.path.join(dataset_root, "train.json")
        json_path = pose_json_path if os.path.exists(pose_json_path) else basic_json_path

        with open(json_path, 'rt', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def _read_rgb(self, path: str):
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        item = self.data[idx]

        target_filename = item.get('gt')
        prompt = item.get('caption', '')
        pose_rel = item.get('pose', None)  # e.g. "train_pose/pose/009292_0.jpg"

        # Resolve paths (dataset_root is a relative string in utils/config.py)
        # Target candidates: dataset_root/<gt>, dataset_root/train/<gt>, dataset_root/images/<gt>
        target_candidates = []
        if target_filename:
            target_candidates.extend([
                os.path.join(dataset_root, target_filename),
                os.path.join(dataset_root, "train", target_filename),
                os.path.join(dataset_root, "images", target_filename),
                os.path.join(dataset_root, "gt", target_filename),
            ])

        pose_path = os.path.join(dataset_root, pose_rel) if pose_rel else None

        # Read pose as hint (if available)
        pose_img = self._read_rgb(pose_path) if pose_path else None

        # Read target image
        target_img = None
        for p in target_candidates:
            target_img = self._read_rgb(p)
            if target_img is not None:
                break

        # If gt is missing, fall back to pose image (keeps training running).
        if target_img is None and pose_img is not None:
            target_img = pose_img

        if target_img is None:
            # Last resort: return a black image to avoid crashing the training loop.
            target_img = np.zeros((512, 512, 3), dtype=np.uint8)
            pose_img = target_img

        if pose_img is None:
            pose_img = target_img

        # Normalize source images to [0, 1].
        source = pose_img.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target_img.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
