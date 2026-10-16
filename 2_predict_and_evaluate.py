import os
import sys

from src.utils import run_standalone_evaluation_pipeline

if __name__ == "__main__":
    run_standalone_evaluation_pipeline(
            checkpoint_path=r"./output/best_bh_unet_model_9CH.pth",      # 可任意填入 4CH, 5CH, 8CH 或 9CH 權重
            image_path=r"./data/PlanetScope_20200518_composite.tif",
            depth_path=r"./data/fine_grid_5m_waterdepth.tif",
            ground_truth_path=r"./data/mask_aligned.tif",
            roi_path=r"./data/roi_10m.geojson",
            preprocess_npz_path=r"./output/test/preprocessed_benthic_data_9CH.npz",
            output_tif=r"./output/evaluation_9CH/habitat_map_pred.tif",
            log_file_path=r"./output/evaluation_9CH/evaluation_report.log",
            batch_size=8
        )