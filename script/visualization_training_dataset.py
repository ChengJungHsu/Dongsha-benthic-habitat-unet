import os
import sys
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import ACA_BENTHIC_CLASSES, SUPERDOVE_8BANDS, ACA_HEX_COLORS

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DFKai-SB', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_npz_inspection_geo(
    npz_path: str,
    ref_sat_tif: str,
    coastline_file: str,
    output_fig: str = "./output/npz_spatial_and_class_distribution_geo.png"
):
    """
    將 NPZ 資料切分以真實 WGS84 經緯度空間、東沙環礁地理範圍與海岸線向量視覺化呈現
    """
    assert os.path.exists(npz_path), f"找不到 NPZ 檔案: {npz_path}"
    assert os.path.exists(ref_sat_tif), f"找不到參考衛星影像: {ref_sat_tif}"
    
    # 1. 讀取參考衛星影像的空間投影與變換矩陣
    with rasterio.open(ref_sat_tif) as src:
        src_crs = src.crs           # 例如 EPSG:32650 (UTM 50N)
        src_transform = src.transform
        height, width = src.height, src.width
        # 計算四角邊界的 UTM 坐標
        minx, miny, maxx, maxy = src.bounds

    # 2. 轉換邊界至 WGS84 (EPSG:4326) 經緯度
    lon_min, lat_min, lon_max, lat_max = transform_bounds(src_crs, "EPSG:4326", minx, miny, maxx, maxy)
    geo_extent = [lon_min, lon_max, lat_min, lat_max]

    # 3. 讀取海岸線向量並統一轉換為 EPSG:4326 (經緯度)
    coastline_gdf = None
    if os.path.exists(coastline_file):
        try:
            coastline_gdf = gpd.read_file(coastline_file)
            coastline_gdf = coastline_gdf.to_crs("EPSG:4326")
            print(">>> 成功載入海岸線向量並轉換為 WGS84 經緯度！")
        except Exception as e:
            print(f"讀取海岸線失敗 ({e})")

    # 4. 讀取 NPZ 快取數據
    data = np.load(npz_path)
    train_x = data['train_x']
    train_y = data['train_y']
    val_x   = data['val_x']
    val_y   = data['val_y']
    val_mask = data['val_blocks_mask']
    class_codes = list(data['class_codes'])  # [1, 2, 3, 4, 5, 6]
    
    # 計算每個 class_code 對應的 0-based 索引 (0, 1, 2, 3, 4, 5)
    train_counts = []
    val_counts = []
    for idx, c in enumerate(class_codes):
        # 切塊內的標籤值是 0-based 的 idx，不是 1-based 的 c
        train_counts.append(int(np.sum(train_y == idx)))
        val_counts.append(int(np.sum(val_y == idx)))

    class_names = [f"[{c}] {ACA_BENTHIC_CLASSES.get(c, str(c))}" for c in class_codes]

    # 5. 繪製圖表
    fig = plt.figure(figsize=(19, 8.5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    # -------------------------------------------------------------
    # 子圖 (A): 真實地理經緯度與海岸線疊加
    # -------------------------------------------------------------
    ax_map = fig.add_subplot(gs[0, 0])
    
    # 繪製紅藍空間獨立區塊 (以 WGS84 extent 顯示)
    cmap_split = mcolors.ListedColormap(['#EEF2FF', '#EF4444'])  # 淺藍/深紅
    ax_map.imshow(
        val_mask.astype(np.uint8),
        extent=geo_extent,
        cmap=cmap_split,
        interpolation='nearest',
        alpha=0.85,
        origin='upper',
        zorder=1
    )

    # 疊加真實海岸線向量
    # if coastline_gdf is not None and not coastline_gdf.empty:
        # coastline_gdf.boundary.plot(ax=ax_map, color='black', linewidth=1.5, zorder=3, label="Coastline (海岸線)")
    
    # 2. ★ 修正：疊加海岸線向量 (加入 aspect=None 防止維度崩潰，並處理幾何型態)
    if coastline_gdf is not None and not coastline_gdf.empty:
        # 過濾空幾何
        valid_coast = coastline_gdf[~coastline_gdf.is_empty & coastline_gdf.geometry.notnull()]
        
        # 判斷是否為多邊形 (Polygon/MultiPolygon)，多邊形取 boundary，線段直接繪製
        geom_types = valid_coast.geom_type.unique()
        if any('Polygon' in t for t in geom_types):
            plot_geom = valid_coast.boundary
        else:
            plot_geom = valid_coast

        plot_geom.plot(
            ax=ax_map,
            color='black',
            linewidth=1.5,
            zorder=3,
            aspect=None  # ★ 關鍵參數：禁止 GeoPandas 自動計算 aspect，徹底解決 ValueError
        )

    # 3. 強制固定視角範圍為影像的真實經緯度範圍
    ax_map.set_xlim(geo_extent[0], geo_extent[1])
    ax_map.set_ylim(geo_extent[2], geo_extent[3])
    
    # 手動維持正確的經緯度視覺長寬比 (以東沙緯度 ~20.7° 計算)
    center_lat = (geo_extent[2] + geo_extent[3]) / 2.0
    ax_map.set_aspect(1.0 / np.cos(np.radians(center_lat)))    
    
    ax_map.set_title("(A) 東沙環礁空間獨立切分分佈 (WGS84 經緯度空間)", fontsize=13, weight='bold', pad=12)
    ax_map.set_xlabel("經度 Longitude (°E)", fontsize=11, weight='bold')
    ax_map.set_ylabel("緯度 Latitude (°N)", fontsize=11, weight='bold')
    ax_map.grid(color='gray', linestyle=':', linewidth=0.6, alpha=0.6)
    
    # 設定經緯度格式小數點 2~3 位
    ax_map.ticklabel_format(useOffset=False, style='plain')

    legend_elements = [
        mpatches.Patch(color='#EEF2FF', label=f"Train Blocks (訓練集 85%, {len(train_x)} 切塊)"),
        mpatches.Patch(color='#EF4444', label=f"Val Blocks (獨立驗證集 15%, {len(val_x)} 切塊)"),
        mpatches.Patch(edgecolor='black', facecolor='none', linewidth=1.5, label="東沙島/環礁海岸線 (Coastline)")
    ]
    ax_map.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.95, facecolor='white')

    # -------------------------------------------------------------
    # 子圖 (B): 類別像元分佈直方圖
    # -------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(class_codes))
    bar_width = 0.38

    ax_bar.bar(x_pos - bar_width/2, train_counts, width=bar_width, 
               label=f"Train Pixels ({sum(train_counts):,})", 
               color="#3B82F6", alpha=0.85, edgecolor="black")
    ax_bar.bar(x_pos + bar_width/2, val_counts, width=bar_width, 
               label=f"Val Pixels ({sum(val_counts):,})", 
               color="#F97316", alpha=0.85, edgecolor="black")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(class_names, rotation=25, ha="right", fontsize=10, weight='bold')
    ax_bar.set_ylabel("有效標籤像元數 (對數尺度 Log Scale)", fontsize=11, weight='bold')
    ax_bar.set_yscale('log')
    ax_bar.set_title("(B) 訓練集 vs. 驗證集 各棲地類別像元分佈", fontsize=13, weight='bold', pad=12)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.6)
    ax_bar.legend(fontsize=10, framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> 空間地理視覺化圖已輸出至: {output_fig}")

if __name__ == "__main__":
    # 檔案路徑設定
    npz_file       = r"../output/dataset/preprocessed_benthic_data_9CH.npz"
    sat_image_path = r"../data/PlanetScope_20200518_composite.tif"
    coastline_path = r"../data/roi_10m.geojson"
    out_png_path   = r"./output/npz_spatial_and_class_distribution_geo_2.png"
    
    plot_npz_inspection_geo(npz_file, sat_image_path, coastline_path, out_png_path)