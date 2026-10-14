import os
import sys
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import polygonize
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import torch
import torch.nn.functional as F
from torch.amp import autocast

# ==============================================================================
# Domain Constants
# ==============================================================================
ACA_BENTHIC_CLASSES = {
    1: "Coral/Algae",
    2: "Seagrass",
    3: "Sand",
    4: "Microalgal Mats",
    5: "Rock",
    6: "Rubble"
}

SUPERDOVE_8BANDS = [
    'coastal_blue', 'blue', 'green_ii', 'green', 
    'yellow', 'red', 'red_edge', 'nir'
]


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
