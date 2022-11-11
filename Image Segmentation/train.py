# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 10:02:05 2022

@author: Saeed Ahmad
"""

import torch
#from albumentations.pytorch import ToTensorV2
from torchvision import transforms as t
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import U_NET



from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
LOAD_MODEL = False
TRAIN_IMG_DIR = "./data/train/"
TRAIN_MASK_DIR = "./data/train_masks/"





def train_fn(loader, model, optimizer, loss_fn):
    #loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loader):
        if batch_idx < 300:
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE, non_blocking=True)

           # print(data.device)

            # forward
            
            predictions = model(data)
            #print("Prediction shape: ", predictions.shape)
            loss = loss_fn(predictions, targets)

            # backward
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update tqdm loop
            #loader.set_postfix(loss=loss.item())



def main():
    train_transform = t.Compose(
        [
            t.ToPILImage(),
            t.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            #A.Rotate(limit=35, p=1.0),
           # A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            t.ToTensor(),
            t.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
           # max_pixel_value=255.0,
        )
           ,
            
        ],
    )

    train_transform_masks = t.Compose(
        [
            t.ToPILImage(),
            t.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            #A.Rotate(limit=35, p=1.0),
           # A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            t.ToTensor(),
            
        ],
    )


    
    

    model = U_NET(in_channels=3, out_channels=1)
    model.to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    train_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        train_transform_masks,
    )


    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(train_loader, model, device=DEVICE)
    #scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print("Training")
        train_fn(train_loader, model, optimizer, loss_fn)

        print("testing")

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(train_loader, model, device=DEVICE)

        # print some examples to a folder
        #save_predictions_as_imgs(
          #  train_loader, model, folder="saved_images/", device=DEVICE
       # )


if __name__ == "__main__":
    main()
