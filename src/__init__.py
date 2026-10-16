"""
Benthic Habitat Mapping using BH-UNet
=====================================
Modular codebase for remote sensing multi-spectral & bathymetric semantic segmentation.
"""

from .model import (
    BHUNet_LateFusion,
    ChannelwiseAttention,
    ConvBlock,
    DecoderBlock,
    DepthFeatureExtractor
)
from .loss import SoftDiceLoss, CombinedLoss, compute_elastic_regularization
from .utils import (
    # 資料管道與前處理
    BenthicSegmentationDataset,
    build_spatial_grid_from_roi,
    prepare_feature_stack,
    align_and_resample_raster,
    convert_linestrings_to_polygons,
    extract_overlapping_patches_fast,
    run_local_preprocessing,
    # 訓練管道
    run_training,
    # 模型結構檢驗與推論評估
    inspect_checkpoint_architecture,
    predict_full_scene_bh_unet,
    log_structured_evaluation,
    run_standalone_evaluation_pipeline,
    # 輔助與常數
    setup_logger,
    ACA_BENTHIC_CLASSES,
    SUPERDOVE_8BANDS    
)

__all__ = [
    # 模型與損失函數
    "BHUNet_LateFusion",
    "ChannelwiseAttention",
    "ConvBlock",
    "DecoderBlock",
    "DepthFeatureExtractor",
    "SoftDiceLoss",
    "CombinedLoss",
    "compute_elastic_regularization",
    # 地理空間特徵同構處理
    "BenthicSegmentationDataset",
    "build_spatial_grid_from_roi",
    "prepare_feature_stack",
    "align_and_resample_raster",
    "convert_linestrings_to_polygons",
    "extract_overlapping_patches_fast",
    "run_local_preprocessing",
    # 訓練
    "run_training",
    # 推論與評估
    "inspect_checkpoint_architecture",
    "predict_full_scene_bh_unet",
    "log_structured_evaluation",
    "run_standalone_evaluation_pipeline",
    # 輔助與常數
    "setup_logger",
    "ACA_BENTHIC_CLASSES",
    "SUPERDOVE_8BANDS"
]
