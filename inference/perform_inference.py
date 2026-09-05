import os
import torch
import sys
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from model.load_model import load_model
from data.load_data import AI4MarsDataset, generate_splits
from data.view_data import visualize_pair, visualize_pred
from config import TRAIN_IMAGES, TRAIN_LABELS, TEST_LABELS
import numpy as np

def perform_inference(model_name, num_images=1, image_dir=None, save_images=False,device=None):
    ds = AI4MarsDataset(TRAIN_IMAGES, TEST_LABELS, testing=True)
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    model, transforms = load_model(model_name)
    # change model so it only has 4 classes for segmentation
    model.classifier[4] = nn.Conv2d(256, 4, kernel_size=(1,1))
    model.to(device)
    model_state = torch.load("checkpoint.pth", map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.eval()

    # load validation split
    train_split, val_split = generate_splits(0.8, 0.2, ds)
    # val_loader = DataLoader(val_split, 2, shuffle=False)
    img, lbl = val_split[8]
    stacked_img = img.unsqueeze(0).repeat(1, 3, 1, 1)
    stacked_img = stacked_img.to(device)

    with torch.no_grad():
        out = model(stacked_img)
        logits = out['out']
        pred = torch.argmax(logits, dim=1)

    print(lbl.unique())

    visualize_pred(img.squeeze(0), lbl.squeeze(0), pred.squeeze(0))

    
def main():
    perform_inference("resnet_50")

if __name__ == "__main__":
    main()

    

