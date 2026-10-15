import os
import sys
import logging
import numpy as np
import rasterio
import datetime
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import polygonize
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from .model import BHUNet_LateFusion
from .loss import CombinedLoss, compute_elastic_regularization


# ==============================================================================
# Domain Constants
# ==============================================================================
# Correct ACA_BENTHIC_CLASSES which has ID=1 for Seagrass 
ACA_BENTHIC_CLASSES = {
    1: "Seagrass",
    2: "Coral/Algae",
    3: "Sand",
    4: "Microalgal Mats",
    5: "Rock",
    6: "Rubble"
}

SUPERDOVE_8BANDS = [
    'coastal_blue', 'blue', 'green-I', 'green', 
    'yellow', 'red', 'red_edge', 'nir'
]

class BenthicSegmentationDataset(Dataset):
    def __init__(self, patches_data, patches_mask=None, is_train=True):
        self.patches_data = patches_data
        self.patches_mask = patches_mask
        self.is_train = is_train

    def __len__(self):
        return len(self.patches_data)

    def __getitem__(self, idx):
        img = self.patches_data[idx].copy()
        if self.patches_mask is not None:
            mask = self.patches_mask[idx].copy()
            if self.is_train:
                if np.random.rand() > 0.5:
                    img = np.flip(img, axis=1).copy()
                    mask = np.flip(mask, axis=0).copy()
                if np.random.rand() > 0.5:
                    img = np.flip(img, axis=2).copy()
                    mask = np.flip(mask, axis=1).copy()
                if np.random.rand() > 0.6:
                    noise = np.random.normal(0, 0.015, img.shape).astype(np.float32)
                    img = np.clip(img + noise, 0.0, 1.0)
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long()
        return torch.from_numpy(img).float()

# 新增至 src/utils.py
def run_local_preprocessing(
    sat_file: str,
    bathy_file: str,
    groundtruth_file: str,
    roi_file: str,
    output_npz: str,
    target_res: float = 3.0,
    z_max: float = 10.0,
    patch_size: int = 256,
    stride: int = 128,
    include_depth: bool = True,
    target_band_indices: list = None
):
    target_dir = os.path.dirname(output_npz)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    print(">>> 階段 1: 讀取向量與定義網格幾何...")
    roi_gdf = gpd.read_file(roi_file)
    with rasterio.open(sat_file) as src_sat:
        base_crs = src_sat.crs
        total_bands = src_sat.count

    if roi_gdf.crs != base_crs:
        roi_gdf = roi_gdf.to_crs(base_crs)
    roi_gdf = convert_linestrings_to_polygons(roi_gdf)

    minx, miny, maxx, maxy = roi_gdf.total_bounds
    target_width = int(np.ceil((maxx - minx) / target_res))
    target_height = int(np.ceil((maxy - miny) / target_res))
    target_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, target_width, target_height)
    target_shape = (target_height, target_width)

    roi_geoms = [(geom, 1) for geom in roi_gdf.geometry if geom.is_valid and not geom.is_empty]
    rasterized_roi = rasterize(roi_geoms, out_shape=target_shape, transform=target_transform, fill=0, dtype=np.uint8)
    inside_roi_mask = (rasterized_roi == 1)

    print(">>> 階段 2: 空間重採樣光譜波段與地形圖層...")
    if target_band_indices is None:
        target_band_indices = list(range(1, total_bands + 1))

    bands_data = []
    for b_idx in target_band_indices:
        b_arr, _ = align_and_resample_raster(sat_file, base_crs, target_transform, target_shape, band_idx=b_idx)
        bands_data.append(np.nan_to_num(b_arr, nan=0.0).astype(np.float32))

    feature_stack = np.stack(bands_data, axis=0)
    op_channels = feature_stack.shape[0]

    depth, depth_nodata = align_and_resample_raster(bathy_file, base_crs, target_transform, target_shape, band_idx=1, resampling=Resampling.cubic)
    labels, _ = align_and_resample_raster(groundtruth_file, base_crs, target_transform, target_shape, band_idx=1, resampling=Resampling.nearest)
    labels = np.nan_to_num(labels, nan=0).astype(np.int32)

    is_depth_invalid = np.isnan(depth) | np.isinf(depth) | (depth <= 0.0)
    if depth_nodata is not None:
        is_depth_invalid |= np.isclose(depth, depth_nodata)
    valid_water_pixels = inside_roi_mask & (~is_depth_invalid) & (depth <= z_max)

    print(f">>> 階段 3: 執行空間平衡正規化 ({op_channels} 個光譜波段)...")
    for c in range(op_channels):
        valid_vals = feature_stack[c][valid_water_pixels]
        min_v = np.percentile(valid_vals, 0.5) if len(valid_vals) > 0 else 0.0
        max_v = np.percentile(valid_vals, 99.5) if len(valid_vals) > 0 else 1.0
        if max_v > min_v:
            clipped = np.clip(feature_stack[c], min_v, max_v)
            feature_stack[c] = (clipped - min_v) / (max_v - min_v)
        else:
            feature_stack[c] = np.zeros_like(feature_stack[c])
        feature_stack[c] = np.nan_to_num(feature_stack[c], nan=0.0)

    if include_depth:
        norm_depth = np.clip(depth / z_max, 0.0, 1.0).astype(np.float32)
        norm_depth = np.nan_to_num(norm_depth, nan=0.0)
        norm_depth[~valid_water_pixels] = 0.0
        norm_depth = np.expand_dims(norm_depth, axis=0)
        final_feature_stack = np.concatenate([feature_stack, norm_depth], axis=0)
        depth_in_channels = 1
    else:
        final_feature_stack = feature_stack
        depth_in_channels = 0

    total_channels = final_feature_stack.shape[0]

    print(">>> 階段 4: 構建標籤遮罩並執行切塊與空間獨立切分...")
    valid_class_codes = sorted(list(ACA_BENTHIC_CLASSES.keys()))
    class_to_idx = {cls: idx for idx, cls in enumerate(valid_class_codes)}
    ignore_index = 255
    dense_mask = np.full(target_shape, ignore_index, dtype=np.uint8)

    trainable_pixels = valid_water_pixels & (np.isin(labels, valid_class_codes))
    class_counts = []
    for cls_code, cls_idx in class_to_idx.items():
        c_mask = trainable_pixels & (labels == cls_code)
        dense_mask[c_mask] = cls_idx
        class_counts.append(np.sum(c_mask))

    class_counts = np.array(class_counts, dtype=np.float32)
    smoothed_weights = np.sqrt(np.sum(class_counts) / (class_counts + 1e-4))
    smoothed_weights = np.clip(smoothed_weights / np.mean(smoothed_weights), 0.6, 3.5).astype(np.float32)

    patches_x, patches_y, coords = extract_overlapping_patches_fast(
        final_feature_stack, dense_mask, patch_size=patch_size, stride=stride, min_valid_ratio=0.05, ignore_index=ignore_index
    )

    block_size = patch_size * 4
    block_ids = (coords[:, 0] // block_size) * 100 + (coords[:, 1] // block_size)
    unique_blocks = np.unique(block_ids)

    np.random.seed(42)
    np.random.shuffle(unique_blocks)
    val_block_count = max(1, int(len(unique_blocks) * 0.15))
    val_blocks = unique_blocks[:val_block_count]

    val_mask = np.isin(block_ids, val_blocks)
    train_mask = ~val_mask

    val_spatial_mask = np.zeros(target_shape, dtype=bool)
    for r, c_idx in coords[val_mask]:
        val_spatial_mask[r:r+patch_size, c_idx:c_idx+patch_size] = True

    np.savez_compressed(
        output_npz,
        train_x=patches_x[train_mask],
        train_y=patches_y[train_mask],
        val_x=patches_x[val_mask],
        val_y=patches_y[val_mask],
        loss_weights=smoothed_weights,
        class_codes=np.array(valid_class_codes),
        val_blocks_mask=val_spatial_mask,
        op_channels=np.array(op_channels),
        depth_in_channels=np.array(depth_in_channels),
        total_channels=np.array(total_channels)
    )

    print(f"特徵輸出統計: 總通道數={total_channels} (光譜={op_channels}, 水深={depth_in_channels})")
    print(f">>> 前處理完成，壓縮包已儲存至: {output_npz}")
    return depth_in_channels, op_channels, total_channels

def run_training(
    data_path,
    output_path,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.0010,
    early_stopping_patience: int = 15,
    depth_in_channels: int = 1,
    depth_channels: int = 32
):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練裝置: {device} | 批次大小: {batch_size}")

    data = np.load(data_path, mmap_mode='r')
    train_x = data['train_x']
    train_y = data['train_y']
    val_x = data['val_x']
    val_y = data['val_y']
    loss_weights = torch.tensor(data['loss_weights'], dtype=torch.float32).to(device)
    num_classes = len(data['class_codes'])
    in_channels = train_x.shape[1]
    op_channels = in_channels - depth_in_channels
    print(f"輸入通道數: {in_channels} | 光譜通道數: {op_channels} | 地形通道數: {depth_in_channels}")

    train_ds = BenthicSegmentationDataset(train_x, train_y, is_train=True)
    val_ds   = BenthicSegmentationDataset(val_x, val_y, is_train=False)

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"訓練切塊: {len(train_ds)} | 獨立驗證切塊: {len(val_ds)}")
    
    model = BHUNet_LateFusion(
        op_channels=op_channels,
        depth_in_channels=depth_in_channels,
        use_late_fusion=flag_use_late_fusion,
        num_classes=num_classes, 
        base_channels=16, 
        depth_channels=depth_channels).to(device)
    
    criterion = CombinedLoss(weight=loss_weights, num_classes=num_classes, ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')
    best_weights_path = output_path
    patience_cnt = 0

    t_start = datetime.datetime.now()
    for epoch in range(1, epochs + 1):
        t_ep = datetime.datetime.now()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                logits = model(x_b)
                ce_loss = criterion(logits, y_b)
                reg_loss = compute_elastic_regularization(model)
                loss = ce_loss + reg_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += ce_loss.item() * x_b.size(0)
            preds = torch.argmax(logits, dim=1)
            valid = (y_b != 255)
            train_correct += torch.sum((preds == y_b) & valid).item()
            train_total += torch.sum(valid).item()

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc = (train_correct / train_total) if train_total > 0 else 0.0

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device, non_blocking=True), y_v.to(device, non_blocking=True)
                with autocast('cuda'):
                    logits_v = model(x_v)
                    loss_v = criterion(logits_v, y_v)
                val_loss += loss_v.item() * x_v.size(0)
                preds_v = torch.argmax(logits_v, dim=1)
                valid_v = (y_v != 255)
                val_correct += torch.sum((preds_v == y_v) & valid_v).item()
                val_total += torch.sum(valid_v).item()

        val_loss /= len(val_loader.dataset)
        val_acc = (val_correct / val_total) if val_total > 0 else 0.0
        sec = (datetime.datetime.now() - t_ep).total_seconds()

        print(f"Epoch [{epoch:03d}/{epochs:03d}] ({sec:.1f}s) - Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.2f}%) | Val Loss: {val_loss:.4f} (Acc: {val_acc*100:.2f}%)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'in_channels': in_channels,
                'num_classes': num_classes
            }, best_weights_path)
        else:
            patience_cnt += 1
            if patience_cnt >= early_stopping_patience:
                print(f"觸發 Early Stopping (連續 {early_stopping_patience} 個 Epoch 未改善)，提前結束。")
                break

    print(f"訓練完成，耗時: {(datetime.datetime.now()-t_start).total_seconds()/60.0:.2f} 分鐘，權重已儲存至: {best_weights_path}")


# ==============================================================================
# Spatial & Raster Utilities
# ==============================================================================
def setup_logger(log_file_path: str, logger_name: str = "BH_UNet") -> logging.Logger:
    """Configures a file and console stream logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def convert_linestrings_to_polygons(gdf: gpd.GeoDataFrame, logger: logging.Logger = None) -> gpd.GeoDataFrame:
    """Auto-detects and closes LineString/MultiLineString geometries to form valid Polygons."""
    new_geoms = []
    converted = False
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            new_geoms.append(geom)
        elif geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 3:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                poly = Polygon(coords)
                if poly.is_valid:
                    new_geoms.append(poly)
                    converted = True
        elif geom.geom_type == 'MultiLineString':
            polys = list(polygonize(geom))
            if polys:
                new_geoms.extend(polys)
                converted = True
            else:
                for line in geom.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 3:
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        poly = Polygon(coords)
                        if poly.is_valid:
                            new_geoms.append(poly)
                            converted = True

    if converted and logger:
        logger.info("檢測到 ROI 為線段，已成功自動重構封閉為多邊形！")
    return gpd.GeoDataFrame(geometry=new_geoms, crs=gdf.crs)


def align_and_resample_raster(
    src_path: str,
    target_crs,
    target_transform,
    target_shape: tuple,
    band_idx: int = 1,
    resampling: Resampling = Resampling.bilinear
):
    """Reprojects and resamples a single raster band to a target grid."""
    target_height, target_width = target_shape
    destination = np.full((target_height, target_width), np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        nodata_val = src.nodata
        reproject(
            source=rasterio.band(src, band_idx),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=resampling,
            src_nodata=nodata_val,
            dst_nodata=np.nan
        )
    return destination, nodata_val


def extract_overlapping_patches_fast(
    stack: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    min_valid_ratio: float = 0.05,
    ignore_index: int = 255
):
    """
    Extracts sliding window patches and records their top-left spatial grid coordinates.
    """
    c, h, w = stack.shape
    patches_x, patches_y, coords = [], [], []

    for r in range(0, h - patch_size + 1, stride):
        for c_idx in range(0, w - patch_size + 1, stride):
            sub_mask = mask[r:r+patch_size, c_idx:c_idx+patch_size]
            valid_pixels = np.sum(sub_mask != ignore_index)
            if valid_pixels / (patch_size * patch_size) >= min_valid_ratio:
                sub_stack = stack[:, r:r+patch_size, c_idx:c_idx+patch_size]
                patches_x.append(sub_stack)
                patches_y.append(sub_mask)
                coords.append((r, c_idx))

    return np.array(patches_x, dtype=np.float32), np.array(patches_y, dtype=np.uint8), np.array(coords)


# ==============================================================================
# Full-Scene Tiling Inference & Metrics
# ==============================================================================
def predict_full_scene_bh_unet(
    model: torch.nn.Module,
    feature_stack: np.ndarray,
    target_shape: tuple,
    num_classes: int,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 128,
    batch_size: int = 8
) -> np.ndarray:
    """
    Performs full-scene sliding window inference with 2D Hanning window smoothing.
    Eliminates border seamline artifacts across overlapping patches.
    """
    model.eval()
    c, h, w = feature_stack.shape

    hann_1d = np.hanning(patch_size)
    hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)
    hann_2d = np.maximum(hann_2d, 1e-4)

    prob_accum = np.zeros((num_classes, h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)

    coords = []
    for r in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            coords.append((r, col))
    if (h - patch_size) % stride != 0:
        for col in range(0, w - patch_size + 1, stride):
            coords.append((h - patch_size, col))
    if (w - patch_size) % stride != 0:
        for r in range(0, h - patch_size + 1, stride):
            coords.append((r, w - patch_size))
    coords.append((h - patch_size, w - patch_size))
    coords = list(set(coords))

    use_cuda = (device.type == 'cuda')
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i+batch_size]
            batch_patches = [feature_stack[:, r:r+patch_size, col:col+patch_size] for r, col in batch_coords]
            batch_tensor = torch.from_numpy(np.array(batch_patches)).float().to(device)

            if use_cuda:
                with autocast('cuda'):
                    logits = model(batch_tensor)
                    probs = F.softmax(logits, dim=1).float().cpu().numpy()
            else:
                logits = model(batch_tensor)
                probs = F.softmax(logits, dim=1).float().cpu().numpy()

            for b_idx, (r, col) in enumerate(batch_coords):
                prob_accum[:, r:r+patch_size, col:col+patch_size] += probs[b_idx] * hann_2d
                weight_accum[r:r+patch_size, col:col+patch_size] += hann_2d

    weight_accum = np.maximum(weight_accum, 1e-6)
    return prob_accum / weight_accum

def log_structured_evaluation(
    logger: logging.Logger,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_dict: dict,
    title_prefix: str = ""
) -> float:
    """Computes and logs Confusion Matrix, OA, Kappa, PA, UA, IoU, and F1-Scores."""
    all_classes = sorted(list(class_dict.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    col_totals = cm.sum(axis=0)
    row_totals = cm.sum(axis=1)

    pa_list, ua_list, f1_list, iou_list = [], [], [], []
    for i in range(len(all_classes)):
        diag = cm[i, i]
        union = row_totals[i] + col_totals[i] - diag
        pa = (diag / row_totals[i]) if row_totals[i] > 0 else 0.0
        ua = (diag / col_totals[i]) if col_totals[i] > 0 else 0.0
        f1 = (2 * pa * ua / (pa + ua)) if (pa + ua) > 0 else 0.0
        iou = (diag / union) if union > 0 else 0.0
        pa_list.append(pa)
        ua_list.append(ua)
        f1_list.append(f1)
        iou_list.append(iou)

    logger.info("=" * 115)
    logger.info(f"                         {title_prefix}混淆矩陣 (ACA 6大底棲類別標準矩陣)")
    logger.info("=" * 115)
    header = f"{'真實類別 (True Class)':<22} | " + " ".join([f"{class_dict.get(c, str(c))[:8]:>8}" for c in all_classes]) + f" | {'總數 (Total)':>10} | {'製圖者精度(PA)':>14}"
    logger.info(header)
    logger.info("-" * 115)
    for i, c in enumerate(all_classes):
        c_name = f"[{c}] {class_dict.get(c, 'Unknown')}"
        row_str = f"{c_name:<22} | " + " ".join([f"{cm[i, j]:>8d}" for j in range(len(all_classes))]) + f" | {row_totals[i]:>10d} | {pa_list[i]*100:>13.2f}%"
        logger.info(row_str)
    logger.info("-" * 115)
    pred_row = f"{'預測總數 (Total Pred)':<22} | " + " ".join([f"{col_totals[j]:>8d}" for j in range(len(all_classes))]) + f" | {cm.sum():>10d} |"
    logger.info(pred_row)
    ua_row = f"{'使用者精度(UA)':<22} | " + " ".join([f"{ua_list[j]*100:>7.2f}%" for j in range(len(all_classes))]) + f" | {'':>10} |"
    logger.info(ua_row)
    logger.info("=" * 115)

    logger.info("" + "=" * 95)
    logger.info(f"                     {title_prefix}各底棲類別精度評估報告 (Classification Metrics)")
    logger.info("=" * 95)
    logger.info(f"{'類別代碼與名稱':<25} | {'測試樣本數':>10} | {'使用者精度 (UA)':>13} | {'製圖者精度 (PA)':>13} | {'IoU':>8} | {'F1-Score':>10}")
    logger.info("-" * 95)
    valid_f1 = []
    for i, c in enumerate(all_classes):
        c_name = f"[{c}] {class_dict.get(c, 'Unknown')}"
        logger.info(f"{c_name:<25} | {row_totals[i]:>10d} | {ua_list[i]*100:>12.2f}% | {pa_list[i]*100:>12.2f}% | {iou_list[i]:>8.4f} | {f1_list[i]:>10.4f}")
        if row_totals[i] > 0:
            valid_f1.append(f1_list[i])
    logger.info("-" * 95)
    logger.info(f"【整體精度 Overall Accuracy (OA)】 : {oa * 100:.2f}% ({oa:.4f})")
    logger.info(f"【一致性檢定 Cohen's Kappa 係數】    : {kappa:.4f}")
    logger.info(f"【宏觀平均 Mean IoU】              : {np.mean(iou_list):.4f}")
    logger.info(f"【宏觀平均 Macro F1-Score】        : {np.mean(valid_f1):.4f}")
    logger.info("=" * 95 + "")

    return oa
