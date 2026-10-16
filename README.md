# PlanetScope SuperDove 多光譜 + 數值水深 (Bathymetry) BH-UNet 底棲棲地分類專案

本專案實作結合高解析度 PlanetScope SuperDove 多光譜衛星遙測影像與數值水深模型（Bathymetry），透過引入通道注意力機制（Channelwise Attention）與延後特徵融合（Late Fusion）的 **BH-UNet** 深度學習架構，針對 Allen Coral Atlas (ACA) 標準 6 大底棲棲地分類體系進行高解析度語意分割（Semantic Segmentation）與全場景空間製圖。

本版本為 **`dev2` 重構更新版**，全面實現「**地理特徵處理數值同構**」、「**4CH / 5CH / 8CH / 9CH 多通道彈性切換**」、「**Late Fusion 與 Early Fusion 自由配置**」以及「**推論端自動架構檢驗**」，大幅強化模型泛化能力與工程架構之模組化程度。

---

## 目錄
- [系統核心特點 (dev2 更新要點)](#系統核心特點-dev2-更新要點)
- [系統架構與協同工作流程](#系統架構與協同工作流程)
- [模型架構：BHUNet_LateFusion](#模型架構bhunet_latefusion)
- [輸入資訊、波段配置與成果輸出](#輸入資訊波段配置與成果輸出)
- [專案結構與模組說明](#專案結構與模組說明)
- [可調控關鍵參數清單](#可調控關鍵參數清單)
- [執行步驟指引 (本地與雲端)](#執行步驟指引-本地與雲端)
- [實驗指標對比與性能分析](#實驗指標對比與性能分析)

---

## 系統核心特點 (dev2 更新要點)

1. **地理特徵處理數值同構 (Processing Isomorphism)**：
   * 抽取共用核心函數 `build_spatial_grid_from_roi` 與 `prepare_feature_stack` 於 `src/utils.py`。
   * 前處理（切塊訓練）與推論出圖（全圖預測）**100% 呼叫完全相同的特徵提取與 0.5%~99.5% 空間平衡分位數正規化邏輯**，消除過去因兩端代碼分立造成的尺度漂移與邊界偏差。
2. **多通道動態相容 (4CH / 5CH / 8CH / 9CH)**：
   * **4CH**：PlanetScope 擷取純光譜 RGB & NIR（Blue: 2, Green: 4, Red: 6, NIR: 8）。
   * **5CH**：4 波段光譜 + 1 通道數值水深。
   * **8CH**：PlanetScope SuperDove 全 8 波段純光譜 。
   * **9CH**：8 波段光譜 + 1 通道數值水深 。
3. **特徵融合策略彈性解耦 (Late Fusion vs. Early Fusion)**：
   * 內建 `BHUNet_LateFusion`，透過 `DepthFeatureExtractor` 將平滑水深獨立降採樣 16 倍，於 Bottleneck 進行語意特徵拼接 。
   * 支援 `use_late_fusion` 參數：設為 `True` 啟用解耦融合（大幅消除水深均值對淺層光譜注意力的干擾）；設為 `False` 則為早期融合（全部通道直接於第 1 層卷積混合） 。
4. **推論端自動架構檢驗 (Automatic Checkpoint Inspection)**：
   * `run_standalone_evaluation_pipeline` 具備 `inspect_checkpoint_architecture` 演算法，直接分析權重檔第一層卷積形狀與模組鍵值，**自動推導出模型所需通道數與融合模式**，無需手動改寫推論代碼 。
5. **空間獨立棋盤切分 (Spatial Block Split)**：
   * 以 $1024 \times 1024$ 像素的大空間網格抽樣劃分獨立驗證區（15%）與訓練區（85%），徹底阻斷 50% 滑窗重疊造成的空間自相關與數據外洩（Data Leakage） 。

---

## 系統架構與協同工作流程

```
[本地端 (Local Workstation)]
  0_local_preprocess.py (簡潔入口腳本)
    ├── 呼叫 src.utils.build_spatial_grid_from_roi (定義 EPSG:32650 與 3.0m 仿射矩陣)
    ├── 呼叫 src.utils.prepare_feature_stack (動態擷取 4CH/8CH，可選串接水深)
    ├── 有效水體遮罩篩選 (0 < depth <= 10.0m)
    ├── 滑窗切塊提取 (256x256, stride=128, 50% 重疊)
    └── 空間棋盤獨立切分 (1024m 區塊) 
         └── 產出: preprocessed_benthic_data_XCH.npz (包含通道中繼資料)
                     │
                     ▼ 上傳至 Google Drive
[雲端平台 (Google Colab - GPU)]
  1_colab_train.py
    ├── 載入 .npz 快取資料
    ├── 構建 BHUNet_LateFusion (自動設定 op_channels, depth_in_channels)
    ├── 複合損失函數 (CombinedLoss: 0.6 * Weighted-CE + 0.4 * Soft-Dice)
    ├── AMP FP16 自動混合精度加速 + CosineAnnealingLR 學習率排程
    └── 儲存最佳權重檔: best_bh_unet_model_XCH.pth (含架構元數據)
                     │
                     ▼ 下載回本地端
[本地端 (Local Workstation)]
  2_predict_and_evaluate.py (通用推論腳本)
    ├── 自動分析 .pth 權重結構 (判定 4/5/8/9 通道與 Late Fusion 開關)
    ├── 呼叫共用特徵準備模組 (保證與前處理數值 100% 同構)
    ├── 全圖 2D Hanning 窗平滑重疊推論 (無縫消除 Patch 邊界拼接痕跡)
    ├── 產出高解析度地理坐標 GeoTIFF (帶有 EPSG:32650 投影)
    └── 嚴格精度交叉報告: 分別輸出「空間獨立測試區」與「全場景參考」混淆矩陣與指標
```

---

## 模型架構：BHUNet_LateFusion

模型核心定義於 `src/model.py`，支援純光譜、Early Fusion 與 Late Fusion 模式 ：

```
[光譜輸入 (4CH 或 8CH)] ──► Enc1 (16) ──► Enc2 (32) ──► Enc3 (64) ──► Enc4 (128) ──► Bottleneck (256)
                                                                                          │
[水深輸入 (1CH)]       ──► DepthFeatureExtractor (4次 Stride=2 卷積, 輸出 32ch) ──────────┘
                                                                                          │ Concat
                                                                                 [融合層: 288 -> 256]
                                                                                          │
[輸出 (6 類底棲棲地)]  ◄── FinalConv ◄── Dec1 (16) ◄── Dec2 (32) ◄── Dec3 (64) ◄── Dec4 (128)
```
 

* **Channelwise Attention (通道注意力)**：在編碼器與解碼器模組中自適應調整通道權重，增強水下透水關鍵光譜特徵 。
* **Group Normalization (群組正規化)**：全面取代傳統 BatchNorm，即便在本地端小批次（Batch Size = 4~8）下依然保持梯度穩定 。
* **DepthFeatureExtractor**：獨立將平滑水深降採樣至 1/16 尺寸，提煉宏觀水深邊界特徵，避免干擾淺層光譜紋理 。

---

## 輸入資訊、波段配置與成果輸出

### 1. Allen Coral Atlas (ACA) 底棲 6 大標準分類體系
* **[1] Seagrass**：海草床
* **[2] Coral/Algae**：活珊瑚與大型藻類複合底質
* **[3] Sand**：沙質沉積物
* **[4] Microalgal Mats**：微藻席 / 底棲微藻
* **[5] Rock**：裸岩基盤 / 礁岩
* **[6] Rubble**：珊瑚碎石 / 礫石

### 2. 衛星影像與地形圖層
* **PlanetScope SuperDove 衛星影像**（8 波段）：
  * `Coastal Blue (1)`、`Blue (2)`、`Green-I (3)`、`Green (4)`、`Yellow (5)`、`Red (6)`、`Red Edge (7)`、`NIR (8)` 。
  * 當配置為 4CH 模式時，系統自動擷取 **`[2, 4, 6, 8]`**（Blue, Green, Red, NIR） 。
* **數值水深模型 (Bathymetry)**：
  * 高解析度淺海水深光柵，水深範圍 $0 \sim 10.0 \ext{m}$，經由三次方內插重採樣至 3m 解析度 。

### 3. 輸出成果
* `preprocessed_benthic_data_XCH.npz`：空間棋盤劃分之訓練/驗證切塊張量包（內含通道元數據與驗證遮罩） 。
* `best_bh_unet_model_XCH_*.pth`：最佳模型權重檔 。
* `habitat_map_pred.tif`：具備 EPSG:32650 投影坐標之底棲棲地分類圖（LZW 壓縮 GeoTIFF） 。
* `evaluation_report.log`：包含空間獨立測試區與全場景之混淆矩陣、OA、Kappa、PA、UA、IoU 及 F1-Score 評估報表 。

### 成果視覺化與評估圖表

#### 1. Predicted Benthic Map
![benthic_habitat_comparison_gt_vs_pred](https://github.com/user-attachments/assets/29fdafa4-95f4-4dff-9a21-a983284923d2)
*圖 1：底棲棲地預測圖與 Ground Truth 對比*

#### 2. Confusion Matrix
![confusion_matrix_strict_spatial_mean_20runs](https://github.com/user-attachments/assets/ab9bcdc0-1988-44df-ac0e-a51df1e7b36a)
*圖 2：嚴格空間劃分 20 次實驗平均混淆矩陣*

#### 3. Evaluation Statistics
![evaluation_strict_spatial_boxplots](https://github.com/user-attachments/assets/15bacdd3-e1d7-4402-a64d-fea115a072e6)
*圖 3：空間獨立驗證評估統計盒狀圖*

#### 4. Training Dataset Distribution
![npz_spatial_and_class_distribution_geo](https://github.com/user-attachments/assets/13aff47b-ec54-463b-b70c-17a87bd9be1e)
*圖 4：訓練資料集空間與類別地理分佈*


---

## 專案結構與模組說明

```
Dongsha_benthic_habitat_unet/
├── data/                                 # 原始地理圖資 (被 .gitignore 排除)
│   ├── PlanetScope_20200518_composite.tif
│   ├── fine_grid_5m_waterdepth.tif
│   ├── mask_aligned.tif
│   └── roi_10m.geojson
├── output/                               # 產出成果目錄
│   ├── preprocessed_benthic_data_9CH.npz
│   ├── best_bh_unet_model_9CH_LF.pth
│   ├── habitat_map_pred.tif
│   └── evaluation_report.log
├── notebooks/
│   └── 1_colab_train.ipynb               # Google Colab 雲端訓練筆記本
├── src/                                  # 核心演算法與資料處理模組庫
│   ├── __init__.py                       # 統一導出公開 API 介面
│   ├── model.py                          # BHUNet_LateFusion, DepthFeatureExtractor, ConvBlock
│   ├── loss.py                           # CombinedLoss (Weighted CE + Soft Dice)
│   └── utils.py                          # 數值同構特徵準備、空間對齊、滑窗推論與精度評估
├── 0_local_preprocess.py                 # 本地端前處理進入點腳本
├── 1_colab_train.py                      # 雲端 GPU 訓練執行腳本
├── 2_predict_and_evaluate.py             # 通用全自動推論出圖與評估腳本
├── environment.yml                       # Conda 環境檔
├── requirements.txt                      # pip 套件清單
├── .gitignore                            # Git 排除設定
└── README.md                             # 專案說明文件
```

---

## 可調控關鍵參數清單

| 腳本 / 模組 | 參數名稱 | 預設值 | 說明與調優指引 |
| :--- | :--- | :---: | :--- |
| `0_local_preprocess.py` | `include_depth` | `True` | 是否將水深堆疊為額外特徵通道（`False` 則為純光譜） 。 |
| | `target_band_indices` | `None` | 指定光譜波段索引。`None` 為讀取全波段；設 `[2, 4, 6, 8]` 擷取 RGB+NIR 。 |
| | `target_res` | `3.0` | 空間重採樣網格解析度（公尺） 。 |
| | `z_max` | `10.0` | 有效透水水深上限（公尺） 。 |
| | `patch_size` / `stride` | `256` / `128` | 切塊大小與移動步長（`stride=128` 即 50% 重疊） 。 |
| `1_colab_train.py` | `flag_use_late_fusion`| `True` | 是否啟用 Late Fusion 解耦融合分支 。 |
| | `batch_size` | `32` | 訓練批次大小（Colab T4 建議設 32） 。 |
| | `lr` | `0.001` | AdamW 初始學習率，隨 Cosine 退火 。 |
| | `epochs` / `early_stopping_patience` | `80` / `15` | 最大訓練輪數與早停容忍度（建議設 15 避免過早中斷） 。 |
| `2_predict_and_evaluate.py` | `batch_size` | `8` | 本地推論批次（Quadro P1000 4GB 顯存建議設 8） 。 |
| | `stride` | `128` | 推論滑窗步長（配合 2D Hanning 窗消除接縫痕跡） 。 |

---

## 執行步驟指引 (本地與雲端)

### 1. 本地端產製前處理資料 (`0_local_preprocess.py`)
打開 `0_local_preprocess.py` 配置需求模式後執行：
```bash
# 產製 9CH 或 4CH 前處理資料
python 0_local_preprocess.py
```
產出的 `.npz` 壓縮檔上傳至 Google Drive 的指定路徑 。

### 2. 雲端訓練模型 (`1_colab_train.py` 或 Notebook)
在 Colab 上掛載 Google Drive 並執行訓練：
```python
from src.utils import run_training

# 範例：啟動 9CH Late Fusion 訓練
run_training(
    data_path="/content/drive/MyDrive/machine_learning/benthic_unet/preprocessed_benthic_data_9CH.npz",
    output_path="/content/drive/MyDrive/machine_learning/benthic_unet/output/best_bh_unet_model_9CH_LF.pth",
    epochs=80,
    batch_size=32,
    lr=0.001,
    early_stopping_patience=15,
    flag_use_late_fusion=True,
    depth_in_channels=1,
    depth_channels=32
)
```
訓練完成後，將最佳權重 `.pth` 下載至本機 `output/` 目錄 。

### 3. 本地端推論出圖與精度交叉評估 (`2_predict_and_evaluate.py`)
程式具備自動辨識機制，直接執行即可：
```bash
python 2_predict_and_evaluate.py
```
程式將自動輸出分類 GeoTIFF 並在終端機與 Log 檔案中輸出混淆矩陣報表 。

---

## 實驗指標對比與性能分析

本專案在相同地理試驗區（東沙環礁淺海）之各模型版本實測成果對比如下[cite: 62, 63, 64]：

| 模型架構版本 | 輸入特徵配置 | 融合策略 | 全場景 Overall Accuracy (OA) | 全場景 Kappa 係數 | 空間獨立測試區 OA | 核心表現判讀 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **4CH 純光譜** | RGB + NIR (4) | 無 | 67.59% | 0.5644 | 64.76% | 基準純光譜模型，運算最快 。 |
| **5CH Early Fusion** | RGB + NIR + Depth (5) | 早期卷積 | *(特徵通道對應修正後)* ~66.5% | ~0.5500 | ~66.0% | 水深平滑數值對可見光波段產生微幅干擾 。 |
| **5CH Late Fusion** | RGB + NIR + Depth (5) | 延後融合 | **66.98%** | **0.5623** | **67.28%** | 解耦架構有效保護光譜特徵，海草 PA 達 79.2% 。 |
| **8CH 純光譜** | SuperDove 全波段 (8) | 無 | 69.57% | 0.5927 | 68.44% | 光譜解析度提升，大幅改善各類別可分度 。 |
| **9CH Early Fusion** | 8 Bands + Depth (9) | 早期卷積 | 67.12% | 0.5556 | 66.85% | 早期融合易因水深均值主導注意力而過早早停。 |
| **9CH Late Fusion** | 8 Bands + Depth (9) | 延後融合 | **72.02%** | **0.6261** | **68.44%** | **全專案最佳精度**，海草與微藻席 PA 大幅拉升 。 |
