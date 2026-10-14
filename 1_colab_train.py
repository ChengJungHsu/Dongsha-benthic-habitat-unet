import os
import sys
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# 確保可引用專案根目錄的 src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import BHUNet
from src.loss import CombinedLoss, compute_elastic_regularization

# Colab / 本地路徑相容配置
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_PATH = "/content/drive/MyDrive/benthic_unet/preprocessed_benthic_data_9CH.npz"
    OUTPUT_DIR = "/content/drive/MyDrive/benthic_unet/output"
except ImportError:
    DATA_PATH = "./output/preprocessed_benthic_data_9CH.npz"
    OUTPUT_DIR = "./output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
#move to src/utils.py
# class BenthicSegmentationDataset(Dataset):
    # def __init__(self, patches_data, patches_mask=None, is_train=True):
        # self.patches_data = patches_data
        # self.patches_mask = patches_mask
        # self.is_train = is_train

    # def __len__(self):
        # return len(self.patches_data)

    # def __getitem__(self, idx):
        # img = self.patches_data[idx].copy()
        # if self.patches_mask is not None:
            # mask = self.patches_mask[idx].copy()
            # if self.is_train:
                # if np.random.rand() > 0.5:
                    # img = np.flip(img, axis=1).copy()
                    # mask = np.flip(mask, axis=0).copy()
                # if np.random.rand() > 0.5:
                    # img = np.flip(img, axis=2).copy()
                    # mask = np.flip(mask, axis=1).copy()
                # if np.random.rand() > 0.6:
                    # noise = np.random.normal(0, 0.015, img.shape).astype(np.float32)
                    # img = np.clip(img + noise, 0.0, 1.0)
            # return torch.from_numpy(img).float(), torch.from_numpy(mask).long()
        # return torch.from_numpy(img).float()

# def run_training(
    # data_path: str = DATA_PATH,
    # output_dir: str = OUTPUT_DIR,
    # epochs: int = 60,
    # batch_size: int = 32,
    # lr: float = 0.0015,
    # early_stopping_patience: int = 8
# ):
    # torch.backends.cudnn.benchmark = True
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"訓練裝置: {device} | 批次大小: {batch_size}")

    # data = np.load(data_path)
    # train_x = data['train_x']
    # train_y = data['train_y']
    # val_x = data['val_x']
    # val_y = data['val_y']
    # loss_weights = torch.tensor(data['loss_weights'], dtype=torch.float32).to(device)
    # num_classes = len(data['class_codes'])
    # in_channels = train_x.shape[1]

    # train_ds = BenthicSegmentationDataset(train_x, train_y, is_train=True)
    # val_ds   = BenthicSegmentationDataset(val_x, val_y, is_train=False)

    # num_workers = 2 if torch.cuda.is_available() else 0
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    # val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # print(f"輸入通道數: {in_channels} | 訓練切塊: {len(train_ds)} | 獨立驗證切塊: {len(val_ds)}")

    # model = BHUNet(in_channels=in_channels, num_classes=num_classes, base_channels=16).to(device)
    # criterion = CombinedLoss(weight=loss_weights, num_classes=num_classes, ignore_index=255)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    # scaler = GradScaler('cuda')

    # best_val_loss = float('inf')
    # best_weights_path = os.path.join(output_dir, "best_bh_unet_model_9CH.pth")
    # patience_cnt = 0

    # t_start = datetime.datetime.now()
    # for epoch in range(1, epochs + 1):
        # t_ep = datetime.datetime.now()
        # model.train()
        # train_loss, train_correct, train_total = 0.0, 0, 0

        # for x_b, y_b in train_loader:
            # x_b, y_b = x_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            # optimizer.zero_grad(set_to_none=True)

            # with autocast('cuda'):
                # logits = model(x_b)
                # ce_loss = criterion(logits, y_b)
                # reg_loss = compute_elastic_regularization(model)
                # loss = ce_loss + reg_loss

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # train_loss += ce_loss.item() * x_b.size(0)
            # preds = torch.argmax(logits, dim=1)
            # valid = (y_b != 255)
            # train_correct += torch.sum((preds == y_b) & valid).item()
            # train_total += torch.sum(valid).item()

        # scheduler.step()
        # train_loss /= len(train_loader.dataset)
        # train_acc = (train_correct / train_total) if train_total > 0 else 0.0

        # model.eval()
        # val_loss, val_correct, val_total = 0.0, 0, 0
        # with torch.no_grad():
            # for x_v, y_v in val_loader:
                # x_v, y_v = x_v.to(device, non_blocking=True), y_v.to(device, non_blocking=True)
                # with autocast('cuda'):
                    # logits_v = model(x_v)
                    # loss_v = criterion(logits_v, y_v)
                # val_loss += loss_v.item() * x_v.size(0)
                # preds_v = torch.argmax(logits_v, dim=1)
                # valid_v = (y_v != 255)
                # val_correct += torch.sum((preds_v == y_v) & valid_v).item()
                # val_total += torch.sum(valid_v).item()

        # val_loss /= len(val_loader.dataset)
        # val_acc = (val_correct / val_total) if val_total > 0 else 0.0
        # sec = (datetime.datetime.now() - t_ep).total_seconds()

        # print(f"Epoch [{epoch:03d}/{epochs:03d}] ({sec:.1f}s) - Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.2f}%) | Val Loss: {val_loss:.4f} (Acc: {val_acc*100:.2f}%)")

        # if val_loss < best_val_loss:
            # best_val_loss = val_loss
            # patience_cnt = 0
            # torch.save({
                # 'model_state_dict': model.state_dict(),
                # 'epoch': epoch,
                # 'val_loss': val_loss,
                # 'in_channels': in_channels,
                # 'num_classes': num_classes
            # }, best_weights_path)
        # else:
            # patience_cnt += 1
            # if patience_cnt >= early_stopping_patience:
                # print(f"觸發 Early Stopping (連續 {early_stopping_patience} 個 Epoch 未改善)，提前結束。")
                # break

    # print(f"訓練完成，耗時: {(datetime.datetime.now()-t_start).total_seconds()/60.0:.2f} 分鐘，權重已儲存至: {best_weights_path}")

run_training(
    data_path: str = DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.0015,
    early_stopping_patience: int = 8
)
