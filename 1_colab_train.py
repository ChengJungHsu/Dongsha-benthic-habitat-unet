# %%
# !git clone -b feature/late-fusion https://github.com/ChengJungHsu/Dongsha-benthic-habitat-unet.git
# %cd Dongsha-benthic-habitat-unet

# %%
import numpy as np
from src.utils import run_training

# %%
# 確保可引用專案根目錄的 src
# sys.path.append(os.path.dirname(os.path.abspath(".")))

# %%
# from src.model import BHUNet
# from src.loss import CombinedLoss, compute_elastic_regularization


# %%
# Colab / 本地路徑相容配置
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_PATH = "/content/drive/MyDrive/machine_learning/benthic_unet/preprocessed_benthic_data_9CH.npz"
    OUTPUT_DIR = "/content/drive/MyDrive/machine_learning/benthic_unet/output"
except ImportError:
    DATA_PATH = "./output/preprocessed_benthic_data_9CH.npz"
    OUTPUT_DIR = "./output"

# %%
#os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
run_training(data_path=DATA_PATH, output_dir= OUTPUT_DIR, epochs= 60, batch_size= 32, lr= 0.001, early_stopping_patience = 15)
