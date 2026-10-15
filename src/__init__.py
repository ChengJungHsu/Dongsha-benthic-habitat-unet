"""
Benthic Habitat Mapping using BH-UNet
=====================================
Modular codebase for remote sensing multi-spectral & bathymetric semantic segmentation.
"""

from .model import BHUNet_LateFusion
from .loss import SoftDiceLoss, CombinedLoss
from .utils import (
    BenthicSegmentationDataset,
    run_local_preprocessing,
    run_training,
    setup_logger,
    convert_linestrings_to_polygons,
    align_and_resample_raster,
    extract_overlapping_patches_fast,
    predict_full_scene_bh_unet,
    log_structured_evaluation,
    ACA_BENTHIC_CLASSES,
    SUPERDOVE_8BANDS    
)

__all__ = [
    "BHUNet_LateFusion",
    "SoftDiceLoss",
    "CombinedLoss",
    "BenthicSegmentationDataset",
    "run_local_preprocessing",
    "run_training",
    "setup_logger",
    "convert_linestrings_to_polygons",
    "align_and_resample_raster",
    "extract_overlapping_patches_fast",
    "predict_full_scene_bh_unet",
    "log_structured_evaluation",
    "ACA_BENTHIC_CLASSES",
    "SUPERDOVE_8BANDS" 
]
