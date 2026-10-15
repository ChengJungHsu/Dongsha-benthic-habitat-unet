import os
import sys
import numpy as np
import rasterio
from rasterio.warp import Resampling
from rasterio.features import rasterize
import geopandas as gpd
import torch

# 引用 src 模組
from src.model import BHUNet_LateFusion
from src.utils import (
    align_and_resample_raster,
    convert_linestrings_to_polygons,
    predict_full_scene_bh_unet,
    log_structured_evaluation,
    setup_logger,
    ACA_BENTHIC_CLASSES
)

def inspect_checkpoint_architecture(ckpt, default_num_classes=6, logger=None):
    """
    自動檢查並解析 .pth 權重檔所對應的通道與融合架構：
    支援 4CH, 5CH, 8CH, 9CH (Early Fusion / Late Fusion)
    """
    state_dict = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
    
    # 檢查是否含有儲存時的元數據
    has_meta = isinstance(ckpt, dict) and ('op_channels' in ckpt or 'in_channels' in ckpt)
    
    # 檢查是否有水深獨立分支 (判斷是否為 Late Fusion)
    has_depth_branch = any(k.startswith('depth_branch.') for k in state_dict.keys())
    
    if has_meta and 'op_channels' in ckpt:
        op_channels = int(ckpt['op_channels'])
        depth_in_channels = int(ckpt.get('depth_in_channels', 1 if has_depth_branch else 0))
        use_late_fusion = bool(ckpt.get('use_late_fusion', has_depth_branch))
        total_in_channels = int(ckpt.get('in_channels', op_channels + depth_in_channels))
    else:
        # 從第一層卷積核反推權重結構
        enc1_in_channels = state_dict['enc1.conv1.weight'].shape[1]
        
        if has_depth_branch:
            # Late Fusion 模式: 第一層卷積是光譜通道數，水深走 depth_branch
            use_late_fusion = True
            op_channels = enc1_in_channels  # 4 或 8
            depth_in_channels = state_dict['depth_branch.net.0.weight'].shape[1] if 'depth_branch.net.0.weight' in state_dict else 1
            total_in_channels = op_channels + depth_in_channels
        else:
            # 無獨立水深分支: Early Fusion 或 純光譜
            use_late_fusion = False
            if enc1_in_channels in [5, 9]:
                # 早期融合模式 (4+1 或 8+1)
                total_in_channels = enc1_in_channels
                depth_in_channels = 1
                op_channels = total_in_channels - depth_in_channels
            else:
                # 純光譜模式 (4 或 8)
                total_in_channels = enc1_in_channels
                op_channels = enc1_in_channels
                depth_in_channels = 0

    if logger:
        logger.info("-" * 65)
        logger.info(f"【模型架構自動辨識結果】:")
        logger.info(f"  * 總輸入通道數 (Total Channels): {total_in_channels}")
        logger.info(f"  * 光譜通道數 (Optical Channels) : {op_channels}")
        logger.info(f"  * 地形通道數 (Depth Channels)   : {depth_in_channels}")
        logger.info(f"  * 融合架構模式 (Late Fusion)    : {'啟用 (Late Fusion)' if use_late_fusion else '停用/純光譜/Early Fusion'}")
        logger.info("-" * 65)

    return op_channels, depth_in_channels, total_in_channels, use_late_fusion, state_dict

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
    logger.info("啟動底棲棲地 UNet 全圖平滑推論與精度評估 (自動適配 4, 5, 8, 9 通道)")
    logger.info(f"權重路徑: {checkpoint_path} | 運算裝置: {device} | 批次大小: {batch_size}")
    logger.info("=" * 95)

    valid_class_codes = sorted(list(ACA_BENTHIC_CLASSES.keys()))
    inv_class_to_idx = {idx: cls for idx, cls in enumerate(valid_class_codes)}

    # 1. 自動檢測 Checkpoint 並構建匹配的模型
    logger.info(">>> 階段 1: 讀取模型權重檔並自動判定網路架構...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    op_channels, depth_in_channels, total_in_channels, use_late_fusion, state_dict = inspect_checkpoint_architecture(
        ckpt, default_num_classes=len(valid_class_codes), logger=logger
    )

    model = BHUNet_LateFusion(
        op_channels=op_channels,
        depth_in_channels=depth_in_channels,
        use_late_fusion=use_late_fusion,
        num_classes=len(valid_class_codes),
        base_channels=16,
        depth_channels=32
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # 2. 空間網格與幾何對齊
    logger.info(">>> 階段 2: 空間幾何對齊與光柵重採樣...")
    roi_gdf = convert_linestrings_to_polygons(gpd.read_file(roi_path))
    with rasterio.open(image_path) as src_sat:
        base_crs = src_sat.crs
        total_sat_bands = src_sat.count
    if roi_gdf.crs != base_crs:
        roi_gdf = roi_gdf.to_crs(base_crs)

    minx, miny, maxx, maxy = roi_gdf.total_bounds
    target_width = int(np.ceil((maxx - minx) / target_res))
    target_height = int(np.ceil((maxy - miny) / target_res))
    target_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, target_width, target_height)
    target_shape = (target_height, target_width)

    rasterized_roi = rasterize([(g, 1) for g in roi_gdf.geometry if g.is_valid], out_shape=target_shape, transform=target_transform, fill=0, dtype=np.uint8)
    inside_roi_mask = (rasterized_roi == 1)

    # 3. 根據 op_channels 自動決定讀取的波段索引
    if op_channels == 4:
        if total_sat_bands >= 8:
            # 8 波段 SuperDove 取 RGB & NIR: [Blue(2), Green(4), Red(6), NIR(8)]
            target_bands = [2, 4, 6, 8]
            logger.info("自動對應: 從 8 波段影像擷取 4 波段 [Blue(2), Green(4), Red(6), NIR(8)]")
        else:
            target_bands = list(range(1, 5))
            logger.info("自動對應: 讀取 4 波段原生影像 [1, 2, 3, 4]")
    elif op_channels == 8:
        target_bands = list(range(1, 9))
        logger.info("自動對應: 讀取全 8 光譜波段 [1 ~ 8]")
    else:
        target_bands = list(range(1, op_channels + 1))
        logger.info(f"自動對應: 讀取前 {op_channels} 個波段")

    bands_data = []
    for b_idx in target_bands:
        b_arr, _ = align_and_resample_raster(image_path, base_crs, target_transform, target_shape, band_idx=b_idx)
        bands_data.append(np.nan_to_num(b_arr, nan=0.0).astype(np.float32))
    feature_stack = np.stack(bands_data, axis=0) # (op_channels, H, W)

    # 讀取水深與真實標籤
    depth, depth_nodata = align_and_resample_raster(depth_path, base_crs, target_transform, target_shape, band_idx=1, resampling=Resampling.cubic)
    labels, _ = align_and_resample_raster(ground_truth_path, base_crs, target_transform, target_shape, band_idx=1, resampling=Resampling.nearest)
    labels = np.nan_to_num(labels, nan=0).astype(np.int32)

    is_depth_invalid = np.isnan(depth) | np.isinf(depth) | (depth <= 0.0)
    if depth_nodata is not None:
        is_depth_invalid |= np.isclose(depth, depth_nodata)
    valid_water_pixels = inside_roi_mask & (~is_depth_invalid) & (depth <= z_max)

    # 4. 光譜平衡正規化
    logger.info(f">>> 階段 3: 執行 {op_channels} 個光譜通道之空間平衡正規化...")
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

    # 5. 判斷是否需要串接水深通道 (4CH/8CH 不串接；5CH/9CH 串接)
    if depth_in_channels > 0:
        logger.info("自動對應: 加入水深特徵作為額外輸入通道...")
        norm_depth = np.clip(depth / z_max, 0.0, 1.0).astype(np.float32)
        norm_depth = np.nan_to_num(norm_depth, nan=0.0)
        norm_depth[~valid_water_pixels] = 0.0
        final_feature_stack = np.concatenate([feature_stack, np.expand_dims(norm_depth, axis=0)], axis=0)
    else:
        logger.info("自動對應: 純光譜模式，不加入水深通道...")
        final_feature_stack = feature_stack

    logger.info(f"最終推論張量維度: {final_feature_stack.shape} (與模型期望吻合)")

    # 6. 全圖平滑推論
    logger.info(f">>> 階段 4: 執行全圖平滑融合推論 (批次: {batch_size}, 步長: {stride})...")
    prob_map = predict_full_scene_bh_unet(
        model=model,
        feature_stack=final_feature_stack,
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

    # 7. 輸出 GeoTIFF
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

    # 8. 精度評估 (獨立空間測試區 vs. 全圖)
    logger.info("\n>>> 階段 5: 執行嚴格交叉精度評估...")
    trainable_pixels = valid_water_pixels & np.isin(labels, valid_class_codes)

    if os.path.exists(preprocess_npz_path):
        prep = np.load(preprocess_npz_path)
        if 'val_blocks_mask' in prep:
            test_pixels = trainable_pixels & prep['val_blocks_mask']
            logger.info("\n" + "=" * 60)
            logger.info(">>> 【1. 嚴格空間獨立測試區 (Unseen Spatial Blocks)】精度評估：")
            logger.info("=" * 60)
            log_structured_evaluation(logger, labels[test_pixels], final_classified_map[test_pixels], ACA_BENTHIC_CLASSES, "[空間獨立測試區] ")

    logger.info("\n" + "=" * 60)
    logger.info(">>> 【2. 全場景參考標籤 (Full Scene Reference)】精度評估：")
    logger.info("=" * 60)
    log_structured_evaluation(logger, labels[trainable_pixels], final_classified_map[trainable_pixels], ACA_BENTHIC_CLASSES, "[全場景參考] ")

if __name__ == "__main__":
    # 使用時只需替換 checkpoint_path 與對應的 output 檔名，其餘皆由程式自動判斷
    run_standalone_evaluation_pipeline(
        checkpoint_path=r"./output/best_bh_unet_model_4CH.pth",      # 可任意填入 4CH, 5CH, 8CH 或 9CH 權重
        image_path=r"./data/PlanetScope_20200518_composite.tif",
        depth_path=r"./data/fine_grid_5m_waterdepth.tif",
        ground_truth_path=r"./data/mask_aligned.tif",
        roi_file=r"./data/roi_10m.geojson",
        preprocess_npz_path=r"./output/test/preprocessed_benthic_data_4CH.npz",
        output_tif=r"./output/habitat_map_pred.tif",
        log_file_path=r"./output/evaluation_report.log",
        batch_size=8
    )