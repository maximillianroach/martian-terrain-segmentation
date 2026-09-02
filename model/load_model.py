import torch
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_MobileNet_V3_Large_Weights
)

def load_model(model_name: str):
    if model_name.lower() not in ("mobilenet", "resnet_50", "resnet_101"):
        raise ValueError("'model_name' should be one of ('mobilenet', 'resnet_50', 'resnet_101')")

    if model_name == "resnet_50":
        model = deeplabv3_resnet50(DeepLabV3_ResNet50_Weights)
        transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    elif model_name == "resnet_101":
        model = deeplabv3_resnet101(DeepLabV3_ResNet101_Weights)
        transforms = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    else:
        model = deeplabv3_mobilenet_v3_large(DeepLabV3_MobileNet_V3_Large_Weights)
        transforms = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    model.eval()

    # warmup run
    _ = model(torch.randn(1,3,520,520))

    return model, transforms


