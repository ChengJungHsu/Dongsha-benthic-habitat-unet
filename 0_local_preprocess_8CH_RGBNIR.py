import os
import sys
import numpy as np
import rasterio
from rasterio.warp import Resampling
from rasterio.features import rasterize
import geopandas as gpd

# 引用 src 模組
from src.utils import (
    align_and_resample_raster,
    convert_linestrings_to_polygons,
    extract_overlapping_patches_fast,
    ACA_BENTHIC_CLASSES
)

def run_local_preprocessing(
    sat_file: str,
    bathy_file: str,
    groundtruth_file: str,
    roi_file: str,
    output_npz: str,
    target_res: float = 3.0,
    z_max: float = 10.0,
    patch_size: int = 256,
    stride: int = 128
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
    bands_data = []
    for b_i in range(1, 9):
        b_idx = b_i if b_i <= total_bands else min(total_bands, b_i)
        b_arr, _ = align_and_resample_raster(sat_file, base_crs, target_transform, target_shape, band_idx=b_idx)
        bands_data.append(np.nan_to_num(b_arr, nan=0.0).astype(np.float32))

    feature_stack = np.stack(bands_data, axis=0) # (8, H, W)
    depth, depth_nodata = align_and_resample_raster(bathy_file, base_crs, target_transform, target_shape, band_idx=1, resampling=Resampling.cubic)
    labels, _ = align_and_resample_raster(groundtruth_file, base_crs, target_transform, target_shape, band_idx=1, resampling=Resampling.nearest)
    labels = np.nan_to_num(labels, nan=0).astype(np.int32)

    is_depth_invalid = np.isnan(depth) | np.isinf(depth) | (depth <= 0.0)
    if depth_nodata is not None:
        is_depth_invalid |= np.isclose(depth, depth_nodata)
    valid_water_pixels = inside_roi_mask & (~is_depth_invalid) & (depth <= z_max)

    print(">>> 階段 3: 執行空間平衡正規化 (8 Bands> R,G,B,NIR)...")
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
    
    # 提取 4 通道: Blue (1), Green-II (3), Red (5), NIR (7)
    rgb_nir_indices = [1, 3, 5, 7]
    feature_stack_4b = feature_stack[rgb_nir_indices, :, :]  # 形狀: (4, H, W)
    
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
        feature_stack, dense_mask, patch_size=patch_size, stride=stride, min_valid_ratio=0.05, ignore_index=ignore_index
    )

    # 空間棋盤區塊劃分 (Spatial Block Split: 1024 像素大單元)
    block_size = patch_size * 4
    block_ids = (coords[:, 0] // block_size) * 100 + (coords[:, 1] // block_size)
    unique_blocks = np.unique(block_ids)

    np.random.seed(42)
    np.random.shuffle(unique_blocks)
    val_block_count = max(1, int(len(unique_blocks) * 0.15))
    val_blocks = unique_blocks[:val_block_count]

    val_mask = np.isin(block_ids, val_blocks)
    train_mask = ~val_mask

    # 產生全圖空間驗證像素遮罩 (供推論階段精確比對)
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
        val_blocks_mask=val_spatial_mask
    )
    print(f">>> 成功提取 {len(patches_x)} 個切塊 (訓練集: {np.sum(train_mask)}, 獨立驗證集: {np.sum(val_mask)})")
    print(f">>> 前處理檔案已儲存至: {output_npz}")

if __name__ == "__main__":
    run_local_preprocessing(
        sat_file=r"./data/PlanetScope_20200518_composite.tif",
        bathy_file=r"./data/fine_grid_5m_waterdepth.tif",
        groundtruth_file=r"./data/mask_aligned.tif",
        roi_file=r"./data/roi_10m.geojson",
        output_npz=r"./output/preprocessed_benthic_data_8CH_RGBNIR.npz",
        target_res=3.0,
        z_max=10.0,
        patch_size=256,
        stride=128
    )
