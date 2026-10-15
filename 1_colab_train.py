import numpy as np
from src.utils import run_training

# Colab / 本地路徑相容配置
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_PATH = "/content/drive/MyDrive/benthic_unet/preprocessed_benthic_data_9CH.npz"
    OUTPUT_DIR = "/content/drive/MyDrive/benthic_unet/output"
except ImportError:
    DATA_PATH = "./output/preprocessed_benthic_data_9CH.npz"
    OUTPUT_DIR = "./output"

#os.makedirs(OUTPUT_DIR, exist_ok=True)

run_training(
    data_path: str = DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.001,
    early_stopping_patience: int = 15
)
