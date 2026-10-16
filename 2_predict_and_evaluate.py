import os
import sys

from src.utils import run_standalone_evaluation_pipeline

if __name__ == "__main__":
    print("#4CH")
    run_standalone_evaluation_pipeline(
            checkpoint_path=r"./output/dataset/best_bh_unet_model_4CH.pth",      # 可任意填入 4CH, 5CH, 8CH 或 9CH 權重
            image_path=r"./data/PlanetScope_20200518_composite.tif",
            depth_path=r"./data/fine_grid_5m_waterdepth.tif",
            ground_truth_path=r"./data/mask_aligned.tif",
            roi_path=r"./data/roi_10m.geojson",
            preprocess_npz_path=r"./output/dataset/preprocessed_benthic_data_5CH.npz",
            output_tif=r"./output/evaluation_4CH/habitat_map_pred.tif",
            log_file_path=r"./output/evaluation_4CH/evaluation_report.log",
            batch_size=8
        )
    print("#5CH_EF")
    run_standalone_evaluation_pipeline(
            checkpoint_path=r"./output/dataset/best_bh_unet_model_5CH_EF.pth",      # 可任意填入 4CH, 5CH, 8CH 或 9CH 權重
            image_path=r"./data/PlanetScope_20200518_composite.tif",
            depth_path=r"./data/fine_grid_5m_waterdepth.tif",
            ground_truth_path=r"./data/mask_aligned.tif",
            roi_path=r"./data/roi_10m.geojson",
            preprocess_npz_path=r"./output/dataset/preprocessed_benthic_data_5CH.npz",
            output_tif=r"./output/evaluation_5CH_EF/habitat_map_pred.tif",
            log_file_path=r"./output/evaluation_5CH_EF/evaluation_report.log",
            batch_size=8
        )
    print("#5CH_LF")
    run_standalone_evaluation_pipeline(
            checkpoint_path=r"./output/dataset/best_bh_unet_model_5CH_LF.pth",      # 可任意填入 4CH, 5CH, 8CH 或 9CH 權重
            image_path=r"./data/PlanetScope_20200518_composite.tif",
            depth_path=r"./data/fine_grid_5m_waterdepth.tif",
            ground_truth_path=r"./data/mask_aligned.tif",
            roi_path=r"./data/roi_10m.geojson",
            preprocess_npz_path=r"./output/dataset/preprocessed_benthic_data_5CH.npz",
            output_tif=r"./output/evaluation_5CH_LF/habitat_map_pred.tif",
            log_file_path=r"./output/evaluation_5CH_LF/evaluation_report.log",
            batch_size=8
        )
    print("#8CH")
    run_standalone_evaluation_pipeline(
            checkpoint_path=r"./output/dataset/best_bh_unet_model_8CH.pth",      # 可任意填入 4CH, 5CH, 8CH 或 9CH 權重
            image_path=r"./data/PlanetScope_20200518_composite.tif",
            depth_path=r"./data/fine_grid_5m_waterdepth.tif",
            ground_truth_path=r"./data/mask_aligned.tif",
            roi_path=r"./data/roi_10m.geojson",
            preprocess_npz_path=r"./output/dataset/preprocessed_benthic_data_8CH.npz",
            output_tif=r"./output/evaluation_8CH/habitat_map_pred.tif",
            log_file_path=r"./output/evaluation_8CH/evaluation_report.log",
            batch_size=8
        )
    print("#9CH_EF")
    run_standalone_evaluation_pipeline(
            checkpoint_path=r"./output/dataset/best_bh_unet_model_9CH_EF.pth",      # 可任意填入 4CH, 5CH, 8CH 或 9CH 權重
            image_path=r"./data/PlanetScope_20200518_composite.tif",
            depth_path=r"./data/fine_grid_5m_waterdepth.tif",
            ground_truth_path=r"./data/mask_aligned.tif",
            roi_path=r"./data/roi_10m.geojson",
            preprocess_npz_path=r"./output/dataset/preprocessed_benthic_data_9CH.npz",
            output_tif=r"./output/evaluation_9CH_EF/habitat_map_pred.tif",
            log_file_path=r"./output/evaluation_9CH_EF/evaluation_report.log",
            batch_size=8
        )
    print("#9CH_LF")
    run_standalone_evaluation_pipeline(
            checkpoint_path=r"./output/dataset/best_bh_unet_model_9CH_LF.pth",      # 可任意填入 4CH, 5CH, 8CH 或 9CH 權重
            image_path=r"./data/PlanetScope_20200518_composite.tif",
            depth_path=r"./data/fine_grid_5m_waterdepth.tif",
            ground_truth_path=r"./data/mask_aligned.tif",
            roi_path=r"./data/roi_10m.geojson",
            preprocess_npz_path=r"./output/dataset/preprocessed_benthic_data_9CH.npz",
            output_tif=r"./output/evaluation_9CH_LF/habitat_map_pred.tif",
            log_file_path=r"./output/evaluation_9CH_LF/evaluation_report.log",
            batch_size=8
        )