import os
import sys

from src.utils import run_local_preprocessing

if __name__ == "__main__":
    sat_file = r"./data/PlanetScope_20200518_composite.tif"
    bathy_file = r"./data/fine_grid_5m_waterdepth.tif"
    groundtruth_file = r"./data/mask_aligned.tif"
    roi_file = r"./data/roi_10m.geojson"
    #8波段+水深(9CH)
    # depth_ch, op_ch, total_ch = run_local_preprocessing(        
        # sat_file, bathy_file, groundtruth_file,  roi_file,
        # output_npz=r"./output/test/preprocessed_benthic_data_9CH.npz",
        # include_depth=True, target_band_indices=None
    # )
    
    #4波段（RGB+NIR）純光譜
    depth_ch, op_ch, total_ch = run_local_preprocessing(        
        sat_file, bathy_file, groundtruth_file,  roi_file,
        output_npz=r"./output/test/preprocessed_benthic_data_4CH.npz",
        include_depth=False, target_band_indices=[2, 4, 6, 8]
    )
    print(f"輸出水深通道數 depth_in_channels = {depth_ch}")
 