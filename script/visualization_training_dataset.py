import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# 取得目前腳本所在目錄 (script/) 的上一層目錄 (即專案根目錄)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 順利引用 src 模組中的常數與函式
from src.utils import ACA_BENTHIC_CLASSES, SUPERDOVE_8BANDS, ACA_HEX_COLORS

def plot_npz_inspection(npz_path: str, output_fig: str = "./output/npz_spatial_and_class_distribution.png"):
    assert os.path.exists(npz_path), f"找不到檔案: {npz_path}"
    print(f">>> 讀取快取檔案: {npz_path}")
    data = np.load(npz_path)

    # 1. 讀取陣列與中繼資訊
    train_x = data['train_x']
    train_y = data['train_y']
    val_x   = data['val_x']
    val_y   = data['val_y']
    class_codes = data['class_codes']
    
    print("-" * 60)
    print(f"訓練集切塊 (train_x): {train_x.shape} | 標籤切塊 (train_y): {train_y.shape}")
    print(f"驗證集切塊 (val_x)  : {val_x.shape}   | 標籤切塊 (val_y)  : {val_y.shape}")
    if 'op_channels' in data and 'depth_in_channels' in data:
        print(f"光譜通道數: {data['op_channels']} | 水深通道數: {data['depth_in_channels']} | 總通道數: {data['total_channels']}")
    print("-" * 60)

    # 2. 統計各類別有效標籤像元數 (排除 255 背景)
    train_counts = [int(np.sum(train_y == c)) for c in class_codes]
    val_counts   = [int(np.sum(val_y == c)) for c in class_codes]
    class_names  = [f"[{c}] {ACA_BENTHIC_CLASSES.get(c, str(c))}" for c in class_codes]

    # 3. 準備畫布 (左圖: 全圖空間棋盤分區；右圖: 類別像元分佈)
    fig = plt.figure(figsize=(18, 8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

    # -------------------------------------------------------------
    # 子圖 1: 空間獨立區塊分佈 (Spatial Blocks Partition)
    # -------------------------------------------------------------
    ax_map = fig.add_subplot(gs[0, 0])
    if 'val_blocks_mask' in data:
        val_mask = data['val_blocks_mask']
        # 0: 訓練區 (淺藍/淺灰), 1: 獨立驗證區 (珊瑚橘/深色)
        cmap_split = mcolors.ListedColormap(['#E0E7FF', '#FF6B6B'])
        ax_map.imshow(val_mask.astype(np.uint8), cmap=cmap_split, interpolation='nearest')

        legend_elements = [
            mpatches.Patch(color='#E0E7FF', label=f"Train Blocks (85%, {len(train_x)} Patches)"),
            mpatches.Patch(color='#FF6B6B', label=f"Val Blocks (15%, {len(val_x)} Patches)")
        ]
        ax_map.legend(handles=legend_elements, loc="upper right", fontsize=11, framealpha=0.95)
        ax_map.set_title("(A) Spatial Block Split (空間棋盤獨立劃分)", fontsize=13, weight='bold')
    else:
        ax_map.text(0.5, 0.5, "未包含 val_blocks_mask 遮罩", ha='center', va='center')
        ax_map.set_title("(A) Spatial Block Split", fontsize=13, weight='bold')
    ax_map.set_xlabel("Pixel Column (X)", fontsize=10)
    ax_map.set_ylabel("Pixel Row (Y)", fontsize=10)

    # -------------------------------------------------------------
    # 子圖 2: 訓練集 vs. 驗證集 各類別像元分佈 (Class Distribution)
    # -------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(class_codes))
    bar_width = 0.38

    bars_train = ax_bar.bar(x_pos - bar_width/2, train_counts, width=bar_width, 
                            label=f"Train Pixels (Total: {sum(train_counts):,})", 
                            color="#3B82F6", alpha=0.85, edgecolor="black")
    bars_val   = ax_bar.bar(x_pos + bar_width/2, val_counts, width=bar_width, 
                            label=f"Val Pixels (Total: {sum(val_counts):,})", 
                            color="#F97316", alpha=0.85, edgecolor="black")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(class_names, rotation=25, ha="right", fontsize=10, weight='bold')
    ax_bar.set_ylabel("Valid Labeled Pixels Count (對數尺度)", fontsize=11, weight='bold')
    ax_bar.set_yscale('log')  # 採用對數坐標軸，方便觀察少數類別 (如 Microalgal Mats)
    ax_bar.set_title("(B) Labeled Pixels Distribution (各棲地類別像元分佈)", fontsize=13, weight='bold')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.6)
    ax_bar.legend(fontsize=10, framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> 空間分區與資料分佈視覺化圖已儲存至: {output_fig}")

if __name__ == "__main__":
    # 請替換為您本地的 .npz 檔案路徑
    npz_file = r"../output/dataset/preprocessed_benthic_data_9CH.npz"
    # npz_file = r"./output/preprocessed_benthic_data_4CH.npz"
    plot_npz_inspection(npz_file)