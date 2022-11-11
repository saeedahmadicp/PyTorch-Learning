# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 10:02:04 2022

@author: Saeed Ahmad
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np




class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, transform_masks=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.transform_mask = transform_masks
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #print(self.image_dir, "mask directory: ", self.images[0][index]) 
        img_path = os.path.join(self.image_dir, str(self.images[index]))
        mask_path = os.path.join(self.mask_dir, str(self.images[index]).replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask[mask == 255.0] = 1.0

      

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform_mask(mask)
           # augmentations = self.transform(image=image, mask=mask)
           # image = augmentations["image"]
           # mask = augmentations["mask"]

        return image, mask








