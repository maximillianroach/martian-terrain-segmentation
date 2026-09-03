import torch
from torch import nn

from torch.utils.data import DataLoader
from data.load_data import generate_splits, AI4MarsDataset
from model.load_model import load_model
import inference.perform_inference
from config import TRAIN_IMAGES, TRAIN_LABELS, TEST_LABELS, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS
import wandb

def train(
        model_name: str,
        batch_size: int,
        num_epochs: int,
        device,
        checkpoint_path="checkpoint.pth",
        # hyperparameters
        lr: float=1e-4, 

):
    # initialize wandb
    wandb.init(
        project="ai4mars-segmentation",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "model_name": model_name
        }
    )
    device = device if device is not None else "cpu"

    # load model
    model, transforms = load_model(model_name)

    # change classifier layer
    model.classifier[4] = nn.Conv2d(256, 4, kernel_size=(1,1))

    model = model.to(device)

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # load data
    ds = AI4MarsDataset(TRAIN_IMAGES, TRAIN_LABELS, testing=False)
    train_split, val_split = generate_splits(0.8, 0.2, dataset=ds)

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0

        # training
        for img, label in train_loader:
            optimizer.zero_grad()
            stacked_img = img.repeat(1, 3, 1, 1)

            stacked_img = stacked_img.to(device)
            label = label.to(device)

            out = model(stacked_img)
            logits = out['out']
            loss = nn.functional.cross_entropy(logits, label, ignore_index=255)
            total_train_loss += loss.item() * img.size(0)

            loss.backward()
            optimizer.step()

        total_val_loss = 0.0
        total_union = torch.zeros(NUM_CLASSES)
        total_intersection = torch.zeros(NUM_CLASSES)

        model.eval()
        with torch.no_grad():
            for img, label in val_loader:
                stacked_img = img.repeat(1, 3, 1, 1)

                stacked_img = stacked_img.to(device)
                label = label.to(device)

                out = model(stacked_img)
                logits = out['out']
                loss = nn.functional.cross_entropy(logits, label, ignore_index=255)
                total_val_loss += loss.item() * img.size(0)

                # IOU
                pred = torch.argmax(logits, dim=1)
                for c in range(NUM_CLASSES):
                    # all the positions where c was predicted
                    pred_c = pred == c
                    # all the positions that correspond to c in label
                    label_c = label == c
                    # number of pixels where class was correctly predicted
                    intersection = (pred_c & label_c).sum().item()
                    # number of pixels where class was predicted
                    union = (pred_c).sum().item() + torch.sum(label_c).sum().item() - intersection

                    total_intersection[c] += intersection
                    total_union[c] += union


        # compute average statistics
        avg_train_loss = total_train_loss / len(train_split)
        avg_val_loss = total_val_loss / len(val_split)
        IoU_per_class = total_intersection / (total_union + 1e-6)
        avg_IoU = IoU_per_class.mean().item()

        print(f"Epoch: {epoch}")
        print(f"train_loss: {avg_train_loss}")
        print(f"val_loss: {avg_val_loss}")
        print(f"avg_IoU: {avg_IoU}")

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "iou_loss": avg_IoU
        })

        # save model when we get new best validation loss
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict, checkpoint_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train("resnet_50", BATCH_SIZE, NUM_EPOCHS, device=device)

if __name__ == "__main__":
    main()