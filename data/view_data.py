import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from data.load_data import AI4MarsDataset
from config import TRAIN_IMAGES, TRAIN_LABELS, TEST_LABELS



label_map = {
    0: (0, 255, 0), # soil - green
    1: (255, 165, 255), # bedrock - yellow
    2: (0, 0, 255), # sand - blue
    3: (255, 0, 0), # big rock - red
    255: (128, 128, 128), # NULL
}

# converts logits to segmentation map
def draw_segmentation_map(outputs):
    # outputs has shape (1, C, H, W)
    # C is class dimension
    
    # take max class along class dim, get model's prediction
    labels = torch.argmax(outputs.squeeze(), dim=0).numpy()

    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    # loops through each class and builds the color region for that class
    for label_num in range(0, len(label_map)):
        # labels has our predictions
        index = labels == label_num

        # get the appropriate colors for this class
        R, G, B = label_map[label_num]

        red_map[index] = R
        green_map[index] = G
        blue_map[index] = B

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def label_to_rgb(labels):
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    # loops through each class and builds the color region for that class
    for label_num in label_map:
        # labels has our predictions
        index = labels == label_num
        index = torch.tensor(index)
        print(index.shape)
        print(red_map.shape)

        # get the appropriate colors for this class
        R, G, B = label_map[label_num]


        red_map[index] = R
        green_map[index] = G
        blue_map[index] = B

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def visualize_pair(img_tensor, label_tensor):
    fig, axes = plt.subplots(1, 2)
    
    labels = label_to_rgb(label_tensor)

    axes[0].imshow(img_tensor.squeeze())

    axes[1].imshow(labels)

    plt.show()

def main():
    ds = AI4MarsDataset(TRAIN_IMAGES, TEST_LABELS, testing=True)
    img_tensor, label_tensor = ds[90]

    visualize_pair(img_tensor, label_tensor)

if __name__ == "__main__":
    main()

