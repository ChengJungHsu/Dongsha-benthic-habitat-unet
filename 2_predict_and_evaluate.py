import os
import sys
import numpy as np
import rasterio
from rasterio.warp import Resampling
from rasterio.features import rasterize
import geopandas as gpd
import torch

# 引用 src 模組
from src.model import BHUNet
from src.utils import (
    align_and_resample_raster,
    convert_linestrings_to_polygons,
    predict_full_scene_bh_unet,
    log_structured_evaluation,
    setup_logger,
    ACA_BENTHIC_CLASSES
)

def run_standalone_evaluation_pipeline(
    checkpoint_path: str,
    image_path: str,
    depth_path: str,
    ground_truth_path: str,
    roi_path: str,
    preprocess_npz_path: str,
    output_tif: str,
    log_file_path: str = "./output/evaluation_report.log",
    target_res: float = 3.0,
    z_max: float = 10.0,
    patch_size: int = 256,
    stride: int = 128,
    batch_size: int = 8
):
    logger = setup_logger(log_file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 95)
    logger.info("啟動 9 通道 (8 Bands + Depth) BH-UNet 全圖平滑推論與精度評估")
    logger.info(f"權重路徑: {checkpoint_path} | 運算裝置: {device} | 批次大小: {batch_size}")
    logger.info("=" * 95)

    valid_class_codes = sorted(list(ACA_BENTHIC_CLASSES.keys()))
    inv_class_to_idx = {idx: cls for idx, cls in enumerate(valid_class_codes)}

    # 1. 載入模型與權重
    logger.info(">>> 階段 1: 載入 BHUNet (in_channels=9)...")
    model = BHUNet(in_channels=9, num_classes=len(valid_class_codes), base_channels=16).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    # 2. 空間網格與圖層重採樣
    logger.info(">>> 階段 2: 空間幾何對齊與光柵重採樣...")
    roi_gdf = convert_linestrings_to_polygons(gpd.read_file(roi_path))
    with rasterio.open(image_path) as src_sat:
        base_crs = src_sat.crs
        total_bands = src_sat.count
    if roi_gdf.crs != base_crs:
        roi_gdf = roi_gdf.to_crs(base_crs)

    minx, miny, maxx, maxy = roi_gdf.total_bounds
    target_width = int(np.ceil((maxx - minx) / target_res))
    target_height = int(np.ceil((maxy - miny) / target_res))
    target_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, target_width, target_height)
    target_shape = (target_height, target_width)

    rasterized_roi = rasterize([(g, 1) for g in roi_gdf.geometry if g.is_valid], out_shape=target_shape, transform=target_transform, fill=0, dtype=np.uint8)
    inside_roi_mask = (rasterized_roi == 1)

    bands_data = []
    for b_i in range(1, 9):
        b_idx = b_i if b_i <= total_bands else min(total_bands, b_i)
        b_arr, _ = align_and_resample_raster(image_path, base_crs, target_transform, target_shape, band_idx=b_idx)
        bands_data.append(np.nan_to_num(b_arr, nan=0.0).astype(np.float32))
    feature_stack = np.stack(bands_data, axis=0)

    depth, depth_nodata = align_and_resample_raster(depth_path, base_crs, target_transform, target_shape, band_idx=1, resampling=Resampling.cubic)
    labels, _ = align_and_resample_raster(ground_truth_path, base_crs, target_transform, target_shape, band_idx=1, resampling=Resampling.nearest)
    labels = np.nan_to_num(labels, nan=0).astype(np.int32)

    is_depth_invalid = np.isnan(depth) | np.isinf(depth) | (depth <= 0.0)
    if depth_nodata is not None:
        is_depth_invalid |= np.isclose(depth, depth_nodata)
    valid_water_pixels = inside_roi_mask & (~is_depth_invalid) & (depth <= z_max)

    # 3. 空間平衡正規化與特徵串接 (8 Bands + 1 Depth = 9 通道)
    logger.info(">>> 階段 3: 光譜與水深特徵正規化並串接為 9 通道...")
    for c in range(8):
        valid_vals = feature_stack[c][valid_water_pixels]
        min_v = np.percentile(valid_vals, 0.5) if len(valid_vals) > 0 else 0.0
        max_v = np.percentile(valid_vals, 99.5) if len(valid_vals) > 0 else 1.0
        if max_v > min_v:
            clipped = np.clip(feature_stack[c], min_v, max_v)
            feature_stack[c] = (clipped - min_v) / (max_v - min_v)
        else:
            feature_stack[c] = np.zeros_like(feature_stack[c])
        feature_stack[c] = np.nan_to_num(feature_stack[c], nan=0.0)

    norm_depth = np.clip(depth / z_max, 0.0, 1.0).astype(np.float32)
    norm_depth = np.nan_to_num(norm_depth, nan=0.0)
    norm_depth[~valid_water_pixels] = 0.0
    feature_stack_9b = np.concatenate([feature_stack, np.expand_dims(norm_depth, axis=0)], axis=0)

    # 4. 全圖平滑推論 (Hanning Window)
    logger.info(f">>> 階段 4: 執行全圖平滑融合推論 (批次: {batch_size}, 步長: {stride})...")
    prob_map = predict_full_scene_bh_unet(
        model=model,
        feature_stack=feature_stack_9b,
        target_shape=target_shape,
        num_classes=len(valid_class_codes),
        device=device,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size
    )

    raw_pred_cls = np.argmax(prob_map, axis=0)
    final_classified_map = np.zeros(target_shape, dtype=np.uint8)
    for idx, cls_code in inv_class_to_idx.items():
        final_classified_map[valid_water_pixels & (raw_pred_cls == idx)] = cls_code

    # 5. 輸出 GeoTIFF
    if output_tif:
        os.makedirs(os.path.dirname(output_tif), exist_ok=True)
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': 0,
            'width': target_width,
            'height': target_height,
            'count': 1,
            'crs': base_crs,
            'transform': target_transform,
            'compress': 'lzw'
        }
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(final_classified_map, 1)
        logger.info(f">>> 成功產製底棲分類圖: {output_tif}")

    # 6. 精度評估 (獨立空間測試區 vs. 全圖)
    logger.info("
>>> 階段 5: 執行嚴格交叉精度評估...")
    trainable_pixels = valid_water_pixels & np.isin(labels, valid_class_codes)

    if os.path.exists(preprocess_npz_path):
        prep = np.load(preprocess_npz_path)
        if 'val_blocks_mask' in prep:
            test_pixels = trainable_pixels & prep['val_blocks_mask']
            logger.info("
" + "=" * 60)
            logger.info(">>> 【1. 嚴格空間獨立測試區 (Unseen Spatial Blocks)】精度評估：")
            logger.info("=" * 60)
            log_structured_evaluation(logger, labels[test_pixels], final_classified_map[test_pixels], ACA_BENTHIC_CLASSES, "[空間獨立測試區] ")

    logger.info("
" + "=" * 60)
    logger.info(">>> 【2. 全場景參考標籤 (Full Scene Reference)】精度評估：")
    logger.info("=" * 60)
    log_structured_evaluation(logger, labels[trainable_pixels], final_classified_map[trainable_pixels], ACA_BENTHIC_CLASSES, "[全場景參考] ")

if __name__ == "__main__":
    run_standalone_evaluation_pipeline(
        checkpoint_path=r"./output/best_bh_unet_model_9CH.pth",
        image_path=r"./data/PlanetScope_20200518_composite.tif",
        depth_path=r"./data/fine_grid_5m_waterdepth.tif",
        ground_truth_path=r"./data/mask_aligned.tif",
        roi_path=r"./data/roi_10m.geojson",
        preprocess_npz_path=r"./output/preprocessed_benthic_data_9CH.npz",
        output_tif=r"./output/habitat_map_superdove8_bh_unet_pred.tif",
        log_file_path=r"./output/evaluation_report.log",
        batch_size=8
    )
