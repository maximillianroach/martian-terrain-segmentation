from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from config import TRAIN_IMAGES, TRAIN_LABELS, TEST_LABELS, IMG_SIZE
import albumentations as A

def build_pairs(img_dir, label_dir, testing=False):
    img_paths = sorted(Path(img_dir).glob("*.JPG"))
    label_paths = sorted(Path(label_dir).glob("*.png"))

    img_stems = {p.stem : p for p in img_paths}

    pairs = []
    missing = []

    # it's easier to loop through the label since there are fewer of them
    for label_path in label_paths:
        # if we are using the test set, the stem ends with _merged, which doesn't occur with training images
        label_stem = label_path.stem if not testing else label_path.stem.split("_merged")[0]
        if label_stem in img_stems:
            # path to img and path to label
            pairs.append((img_stems[label_stem], label_path))
        else:
            # if the label doesn't have a matching image, put in missing
            missing.append(label_path)
    return pairs
        
class AI4MarsDataset(Dataset):
    def __init__(self, img_dir, label_dir, testing=False):
        self.pairs = build_pairs(img_dir, label_dir, testing=testing)

    # tells us how many images are in the dataset
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]

        # resize img and convert it to tensor
        img = Image.open(img_path).convert("L")
        transform = transforms.ToTensor()
        resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
        resized_img = resize(img)
        img_tensor = transform(resized_img)

        # resize label and convert it to tensor
        label_resize = transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST)
        label = Image.open(label_path).convert("L")
        resized_label = label_resize(label)
        label_tensor = torch.from_numpy(np.array(resized_label))

        return (img_tensor, label_tensor)

def generate_splits(train_split, val_split, dataset=AI4MarsDataset(TRAIN_IMAGES,TRAIN_LABELS,testing=False), seed=42):
    train_set, val_set = random_split(dataset, [train_split, val_split],generator=torch.Generator().manual_seed(seed))
    return (train_set, val_set)

def determine_class_counts():
    ds = AI4MarsDataset(TRAIN_IMAGES,TRAIN_LABELS,testing=False)
    loader = DataLoader(ds)

    counts = torch.zeros((4, ))
    # 0 - soil
    # 1 - bedrock
    # 2 - sand
    # 3 big rock

    for img, lbl in loader:
        for label_num in range(0, 4):
            # labels has our predictions
            counts[label_num] += torch.sum(lbl == label_num)

    return counts

def main():
    counts = determine_class_counts()
    total_pixels = torch.sum(counts)
    frequencies = counts / total_pixels
    weights = 1 / frequencies
    print(weights)


if __name__ == "__main__":
    main()
