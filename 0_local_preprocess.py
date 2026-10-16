import os
import sys

from src.utils import run_local_preprocessing

if __name__ == "__main__":
    sat_file = r"./data/PlanetScope_20200518_composite.tif"
    bathy_file = r"./data/fine_grid_5m_waterdepth.tif"
    groundtruth_file = r"./data/mask_aligned.tif"
    roi_file = r"./data/roi_10m.geojson"
    #9CH:8波段+水深
    depth_ch, op_ch, total_ch = run_local_preprocessing(        
        sat_file, bathy_file, groundtruth_file,  roi_file,
        output_npz=r"./output/dataset/preprocessed_benthic_data_9CH.npz",
        include_depth=True, target_band_indices=None
    )
    print(f"輸出水深通道數 depth_in_channels = {depth_ch}")
    #8CH:8波段
    depth_ch, op_ch, total_ch = run_local_preprocessing(        
        sat_file, bathy_file, groundtruth_file,  roi_file,
        output_npz=r"./output/dataset/preprocessed_benthic_data_8CH.npz",
        include_depth=False, target_band_indices=None
    )
    print(f"輸出水深通道數 depth_in_channels = {depth_ch}")
    #5CH:4波段（RGB+NIR)+水深
    depth_ch, op_ch, total_ch = run_local_preprocessing(        
        sat_file, bathy_file, groundtruth_file,  roi_file,
        output_npz=r"./output/dataset/preprocessed_benthic_data_5CH.npz",
        target_res=3.0, z_max=10.0, patch_size=256, stride=128,
        include_depth=True, target_band_indices=[2,4,6,8]
    )
    print(f"輸出水深通道數 depth_in_channels = {depth_ch}")
    #4CH:4波段（RGB+NIR)
    depth_ch, op_ch, total_ch = run_local_preprocessing(        
        sat_file, bathy_file, groundtruth_file,  roi_file,
        output_npz=r"./output/dataset/preprocessed_benthic_data_4CH.npz",
        target_res=3.0, z_max=10.0, patch_size=256, stride=128,
        include_depth=False, target_band_indices=[2,4,6,8]
    )
    print(f"輸出水深通道數 depth_in_channels = {depth_ch}")
 
