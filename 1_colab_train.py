import os
import numpy as np
from src.utils import run_training

# Colab / 本地路徑相容配置
from google.colab import drive
drive.mount('/content/drive')
OUTPUT_DIR = "/content/drive/MyDrive/machine_learning/benthic_unet"
DATA_PATH = os.path.join(OUTPUT_DIR, "preprocessed_benthic_data_4CH.npz")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, output, "best_bh_unet_model_4CH.pth")
#os.makedirs(OUTPUT_DIR, exist_ok=True)

run_training(
    data_path = DATA_PATH,
    output_dir = OUTPUT_PATH,
    epochs = 60,
    batch_size = 32,
    lr = 0.001,
    early_stopping_patience = 15
)
