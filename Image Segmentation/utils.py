# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 10:02:05 2022

@author: Saeed Ahmad
"""

import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    batch_size,
    train_transform,
    train_transform_masks

):



    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        transform_masks = train_transform_masks
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )


    return train_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if idx >= 300:
                x = x.to(device)
                y = y.to(device) #.unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / ( (preds + y).sum() + 1e-8)

    print(
         f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
     )
    print(f"Dice score: {dice_score}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx >= 300:
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()