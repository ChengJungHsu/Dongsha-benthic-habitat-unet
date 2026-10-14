"""
Benthic Habitat Mapping using BH-UNet
=====================================
Modular codebase for remote sensing multi-spectral & bathymetric semantic segmentation.
"""

from .model import BHUNet, ChannelwiseAttention, ConvBlock, DecoderBlock
from .loss import SoftDiceLoss, CombinedLoss, compute_elastic_regularization
from .utils import (
    align_and_resample_raster,
    convert_linestrings_to_polygons,
    extract_overlapping_patches_fast,
    predict_full_scene_bh_unet,
    log_structured_evaluation,
    ACA_BENTHIC_CLASSES,
    SUPERDOVE_8BANDS
)

__all__ = [
    "BHUNet",
    "ChannelwiseAttention",
    "ConvBlock",
    "DecoderBlock",
    "SoftDiceLoss",
    "CombinedLoss",
    "compute_elastic_regularization",
    "align_and_resample_raster",
    "convert_linestrings_to_polygons",
    "extract_overlapping_patches_fast",
    "predict_full_scene_bh_unet",
    "log_structured_evaluation",
    "ACA_BENTHIC_CLASSES",
    "SUPERDOVE_8BANDS"
]
