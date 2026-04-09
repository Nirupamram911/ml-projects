import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class FireDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])

        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (128, 128))
        mask = mask / 255.0

        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)