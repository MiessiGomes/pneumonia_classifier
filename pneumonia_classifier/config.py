import os
from pathlib import Path

# --- Project Structure ---
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data" / "chest_xray"
OUTPUT_DIR: Path = BASE_DIR / "output"
MODEL_DIR: Path = OUTPUT_DIR / "models"
REPORTS_DIR: Path = OUTPUT_DIR / "reports"

# --- Dataset Paths ---
TRAIN_DIR: Path = DATA_DIR / "train"
VAL_DIR: Path = DATA_DIR / "val"
TEST_DIR: Path = DATA_DIR / "test"

# --- Model & Training Parameters ---
IMG_HEIGHT: int = 128
IMG_WIDTH: int = 128
BATCH_SIZE: int = 32
EPOCHS: int = 100
LEARNING_RATE: float = 0.001
MODEL_NAME: str = "pneumonia_classifier.h5"
MODEL_PATH: Path = MODEL_DIR / MODEL_NAME

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
