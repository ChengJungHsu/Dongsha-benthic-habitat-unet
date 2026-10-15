####!git clone https://github.com/ChengJungHsu/Dongsha-benthic-habitat-unet.git
####%cd Dongsha-benthic-habitat-unet.git
import os
import numpy as np
from src.utils import run_training

# Colab / 本地路徑相容配置
from google.colab import drive
drive.mount('/content/drive')
OUTPUT_DIR = "/content/drive/MyDrive/machine_learning/benthic_unet"
DATA_PATH = os.path.join(OUTPUT_DIR, "preprocessed_benthic_data_9CH.npz")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output", "best_bh_unet_model_9CH.pth")
OUTPUT_PATH_LF = os.path.join(OUTPUT_DIR, "output", "best_bh_unet_model_9CH_LF.pth")
#os.makedirs(OUTPUT_DIR, exist_ok=True)

# LateFusion
run_training(
    data_path = DATA_PATH,
    output_path = OUTPUT_PATH_LF,
    epochs = 80,
    batch_size = 32,
    lr = 0.001,
    early_stopping_patience = 15,
    flag_use_late_fusion = True,
    depth_in_channels = 1,
    depth_channels: int = 32
)

# no LateFusion
run_training(
    data_path = DATA_PATH,
    output_path = OUTPUT_PATH,
    epochs = 80,
    batch_size = 32,
    lr = 0.001,
    early_stopping_patience = 15,
    flag_use_late_fusion = False,
    depth_in_channels = 1,
    depth_channels: int = 32
)