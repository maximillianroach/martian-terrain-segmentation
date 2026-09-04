from pathlib import Path

PROJECT_ROOT = Path.cwd()

DATA_ROOT = PROJECT_ROOT / "data" / "ai4mars-dataset-merged-0.6" / "msl" / "ncam"
TRAIN_IMAGES = DATA_ROOT / "images" / "edr"
TRAIN_LABELS = DATA_ROOT / "labels" / "train"
TEST_LABELS = DATA_ROOT / "labels" / "test" / "masked-gold-min3-100agree"

NUM_CLASSES = 4
BATCH_SIZE = 8
IMG_SIZE = 512
NUM_EPOCHS = 10