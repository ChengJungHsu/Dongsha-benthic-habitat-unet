# PlanetScope SuperDove 8-Band + Bathymetry BH-UNet 底棲棲地分類專案

本專案實作結合高解析度 PlanetScope SuperDove 8 波段光學多光譜遙測衛星影像與數值水深模型（Bathymetry），透過引入通道注意力機制（Channelwise Attention）的 **BH-UNet** 深度學習模型，針對 Allen Coral Atlas (ACA) 標準 6 大底棲棲地分類體系進行語意分割（Semantic Segmentation）與全場景製圖。

本專案採用 **「本地端前處理 $\to$ 雲端 GPU 高速訓練 $\to$ 本地端平滑融合推論與精度評估」** 的混合協同工作流程，兼顧大尺度遙測空間資料處理效率與深度學習運算效能。

---

## 目錄
- [系統架構與工作流程](#系統架構與工作流程)
- [核心模型特性 (BH-UNet)](#核心模型特性-bh-unet)
- [輸入資訊與成果輸出](#輸入資訊與成果輸出)
- [專案結構與腳本說明](#專案結構與腳本說明)
- [可調控關鍵參數](#可調控關鍵參數)
- [本地端與 GitHub 開發指引](#本地端與-github-開發指引)

---

## 系統架構與工作流程

```
[本地端 (Local Workstation)]
  0_local_preprocess.py
    ├── 1. 空間幾何網格對齊 (重構多邊形 ROI，定義基準坐標系 EPSG:32650 與 3.0 m 解析度)
    ├── 2. PlanetScope 8 光譜波段與數值水深 (Bathymetry) 空間重採樣
    ├── 3. 有效水體遮罩篩選 (0 < depth <= 10.0 m，排除陸地與深水無效像元)
    ├── 4. 空間平衡分位數正規化 (8 Bands: 0.5%~99.5%, Depth: 0~10m 縮放) -> 堆疊為 9 通道張量
    ├── 5. 50% 重疊滑窗切塊 (256x256, stride=128)
    └── 6. 空間棋盤塊切分 (Spatial Block Split: 1024x1024 空間塊，85% Train / 15% Val，阻斷鄰近外洩)
         └── 產出: preprocessed_benthic_data_9CH.npz (上傳至雲端儲存)
                     │
                     ▼
[雲端平台 (Google Colab / Cloud GPU)]
  1_colab_train.py
    ├── 1. 載入 preprocessed_benthic_data_9CH.npz
    ├── 2. 構建 9 通道 BH-UNet 模型 (in_channels=9, num_classes=6)
    ├── 3. 複合損失函數 (CombinedLoss: 0.6 * Weighted-CE + 0.4 * Soft-Dice)
    ├── 4. 自動混合精度 (AMP FP16) + CosineAnnealing 學習率退火 + 彈性網路正則化 (Elastic Net)
    ├── 5. 空間獨立驗證集監控 (Early Stopping Patience=8)
    └── 6. 回寫最佳權重檔: best_bh_unet_model_9CH.pth (下載回本地端)
                     │
                     ▼
[本地端 (Local Workstation / Dedicated GPU)]
  2_predict_and_evaluate.py
    ├── 1. 載入 best_bh_unet_model_9CH.pth (配置 in_channels=9)
    ├── 2. 讀取原始圖資，以相同標準正規化構建全場景 9 通道空間陣列
    ├── 3. Hanning 窗平滑加權重疊推論 (消除 Patch 拼接邊界效應)
    ├── 4. 輸出標準 GeoTIFF (帶有 EPSG:32650 投影資訊與仿射變換矩陣)
    └── 5. 嚴格精度交叉評估: 分別統計「空間獨立測試區 (Unseen Blocks)」與「全場景參考」混淆矩陣
```

---

## 核心模型特性 (BH-UNet)

1. **通道注意力機制 (Channelwise Attention)**：
   * 在每個編碼器與解碼器特徵提取模組整合 Squeeze-and-Excitation 機制，透過全局平均池化（GAP）與全連接層學習 9 個通道（8 個光譜波段 + 1 個地形水深）的自適應權重，動態增強水下有效穿透波段與地形邊界特徵，抑制水體雜訊。
2. **群組正規化 (GroupNorm)**：
   * 全網路以 `GroupNorm(num_groups=8)` 取代傳統 `BatchNorm`。正規化統計量在通道群組內計算，不依賴批次大小（Batch Size），即使在本地小顯存或大批次環境下均能維持梯度高度穩定。
3. **複合損失函數抗不平衡 (Combined Loss: Weighted CE + Soft Dice)**：
   * 結合以樣本頻率逆平方根計算的 `Class-Weighted CrossEntropy` 與 `SoftDiceLoss`，既保留 Cross-Entropy 的平滑收斂特性，又利用 Dice 損失強化對微藻席（Microalgal Mats）、珊瑚礁碎屑（Rubble）等稀少類別的邊界約束。
4. **Hanning 窗空間平滑融合 (Hanning Window Tiling Fusion)**：
   * 推論階段採用 50% 重疊滑窗推論，並於重疊像元乘載 2D Hanning 窗權重進行平滑累加平均，徹底解決遙測大圖拼貼常見的格網方塊接縫（Seamline Artifacts）。
5. **空間獨立區塊驗證 (Spatial Block Cross-Validation)**：
   * 前處理切塊時以 $1024 \times 1024$ 像素之大空間網格進行訓練集與驗證集劃分，避免相鄰 Patch 因 50% 重疊產生空間自相關（Spatial Autocorrelation）與資料外洩（Data Leakage）。

---

## 輸入資訊與成果輸出

### 1. ACA 底棲棲地 6 大分類體系
* **[1] Coral/Algae**：活珊瑚與大型藻類複合基質
* **[2] Seagrass**：海草床
* **[3] Sand**：沙質沉積物
* **[4] Microalgal Mats**：微藻席 / 底棲微藻
* **[5] Rock**：堅硬岩石基盤 / 礁岩
* **[6] Rubble**：破碎珊瑚石 / 礫石

### 2. 輸入圖資規範
* **多光譜衛星影像**：PlanetScope SuperDove 8-Band 反射率光柵（Coastal Blue, Blue, Green II, Green, Yellow, Red, Red Edge, NIR）。
* **數值水深光柵 (Bathymetry)**：高解析度淺海測深模型（單位：公尺），作為第 9 特徵通道。
* **真實標籤圖層 (Ground Truth)**：對齊 ACA 6 大類別代碼之真實標記光柵。
* **分析範圍向量 (ROI)**：研究區 GeoJSON / Shapefile，支援線段或多邊形幾何。

### 3. 產出成果
* `preprocessed_benthic_data_9CH.npz`：空間切分完成之 9 通道訓練與驗證切塊張量快取包。
* `best_bh_unet_model_9CH.pth`：經空間驗證集篩選之最佳 PyTorch 模型權重檔。
* `habitat_map_superdove8_bh_unet_pred.tif`：具備投影坐標之高解析度底棲棲地分類圖（LZW 壓縮 GeoTIFF）。
* `habitat_map_superdove8_bh_unet_evaluation_report.log`：獨立驗證區與全圖之混淆矩陣、OA、Kappa、PA、UA、IoU 及 F1-Score 評估報表。

---

## 專案結構與腳本說明

```
benthic-habitat-unet/
├── data/                                 # 原始圖資 (被 .gitignore 排除)
│   ├── PlanetScope_20200518_composite.tif
│   ├── fine_grid_5m_waterdepth.tif
│   ├── mask_aligned.tif
│   └── roi_10m.geojson
├── output/                               # 產出成果目錄
│   ├── preprocessed_benthic_data_9CH.npz
│   ├── best_bh_unet_model_9CH.pth
│   ├── habitat_map_superdove8_bh_unet_pred.tif
│   └── habitat_map_superdove8_bh_unet_evaluation_report.log
├── notebooks/
│   └── 1_colab_train.ipynb               # Google Colab 訓練用 Jupyter 筆記本
├── src/
│   ├── __init__.py
│   ├── model.py                          # BHUNet 與 ChannelwiseAttention 架構定義
│   ├── loss.py                           # SoftDiceLoss 與 CombinedLoss 實作
│   └── utils.py                          # 空間幾何對齊、Hanning窗融合與精度評估工具
├── 0_local_preprocess.py                 # 本地端光柵對齊、水深正規化與空間切塊腳本
├── 1_colab_train.py                      # 雲端 GPU 混合精度訓練腳本
├── 2_predict_and_evaluate.py             # 本地端 9 通道全圖平滑推論與精度評估腳本
├── environment.yml                       # Conda 環境配置檔
├── requirements.txt                      # pip 套件清單
├── .gitignore                            # Git 忽略設定
└── README.md                             # 專案說明文件
```

---

## 可調控關鍵參數

| 腳本名稱 | 參數名稱 | 預設值 | 說明與調優指引 |
| :--- | :--- | :--- | :--- |
| `0_local_preprocess.py` | `target_res` | `3.0` | 空間重採樣網格解析度（公尺），建議維持衛星原生解析度。 |
| | `z_max` | `10.0` | 水深截斷上限（公尺），超過此深度之水體視為深海無效像元。 |
| | `patch_size` / `stride` | `256` / `128` | 切塊大小與移動步長。`stride=128` 即 50% 重疊；欲擴大訓練量可縮小至 `64`。 |
| | `block_size` | `1024` | 空間棋盤劃分塊大小，確保訓練與驗證 Patch 間存在物理隔離。 |
| | `min_valid_ratio` | `0.05` | 單張 Patch 中有效標籤像素比例門檻，低於此值則剔除。 |
| `1_colab_train.py` | `in_channels` | `9` | 模型輸入維度（8 個光譜波段 + 1 個數值水深通道）。 |
| | `batch_size` | `32` | 訓練批次大小。Colab T4 (16GB) 建議設 `32`，A100 可設 `64`。 |
| | `lr` | `0.0015` | AdamW 初始學習率，搭配 CosineAnnealing 進行動態退火。 |
| | `epochs` / `early_stopping_patience`| `60` / `8` | 最大訓練輪數與早停耐心值，以驗證集 Loss 為基準。 |
| | `CombinedLoss 權重` | `0.6*CE + 0.4*Dice` | CE 維持宏觀梯度穩定，Dice 提升少數類別分割效果。 |
| `2_predict_and_evaluate.py`| `batch_size` | `8` | 本地推論批次。Quadro P1000 (4GB) 建議設 `8`（推論耗時約 60~80s）。 |
| | `stride` | `128` | 推論移動步長。配合 50% 重疊之 Hanning 窗消除接縫。 |

---

## 本地端與 GitHub 開發指引

### 1. 本地環境建置 (Conda)
建議於本地端工作站建立專屬 Python 虛擬環境：
```bash
conda create -n benthic_ml python=3.10 -y
conda activate benthic_ml

# 安裝 PyTorch (CUDA 11.8 / 12.1 依本地顯卡驅動選擇)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安裝空間光柵與地理資訊套件
conda install -c conda-forge rasterio geopandas shapely scikit-learn pandas
```

### 2. 執行流程
1. **執行本地端前處理**：
   ```bash
   python 0_local_preprocess.py
   ```
   產出 `preprocessed_benthic_data_9CH.npz`。
2. **上傳並於雲端訓練**：
   將 `.npz` 上傳至 Google Drive 指定路徑，開啟 `1_colab_train.py`（或 Notebook）完成訓練，並下載 `best_bh_unet_model_9CH.pth` 回本地 `output/` 目錄。
3. **執行本地端推論與精度評估**：
   ```bash
   python 2_predict_and_evaluate.py
   ```
   檢視終端機精度報表，並可將產出之 `habitat_map_superdove8_bh_unet_pred.tif` 拖曳至 QGIS / ArcGIS 中疊加檢驗。
