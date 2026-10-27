# PlanetScope SuperDove Multispectral + Bathymetry BH-UNet Benthic Habitat Classification

This project implements **BH-UNet**, an advanced deep learning framework incorporating channel-wise attention mechanisms and late feature fusion, designed for semantic segmentation and high-resolution spatial mapping of shallow reef ecosystems. The pipeline integrates high-resolution PlanetScope SuperDove multispectral satellite imagery with numerical bathymetry models (DEM) to classify benthic environments under the standard Allen Coral Atlas (ACA) 6-class benthic classification scheme.

This repository features the **`dev2` refactored release**, introducing end-to-end numerical isomorphism in spatial preprocessing, dynamic multichannel switching (4CH / 5CH / 8CH / 9CH), decoupled fusion configurations (Late Fusion vs. Early Fusion), and automated checkpoint architecture inspection.

---

## Table of Contents
- [Key Features (`dev2` Release)](#key-features-dev2-release)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Model Architecture: BHUNet_LateFusion](#model-architecture-bhunet_latefusion)
- [Data Specifications, Band Configs & Outputs](#data-specifications-band-configs--outputs)
- [Repository Structure](#repository-structure)
- [Key Configurable Parameters](#key-configurable-parameters)
- [Execution Guide (Local & Cloud)](#execution-guide-local--cloud)
- [Experimental Results & Benchmark Analysis](#experimental-results--benchmark-analysis)

---

## Key Features (`dev2` Release)

1. **Numerical Feature Processing Isomorphism**:
   * Centralizes spatial processing in `build_spatial_grid_from_roi` and `prepare_feature_stack` within `src/utils.py`.
   * Patch-based training preprocessing and full-scene inference pipelines share identical feature extraction and robust $0.5\% \sim 99.5\%$ spatial percentile normalization, eliminating domain shifts and edge discrepancies caused by disparate codebases.
2. **Dynamic Multichannel Support (4CH / 5CH / 8CH / 9CH)**:
   * **4CH**: Optical-only standard bands: Blue (2), Green (4), Red (6), and NIR (8).
   * **5CH**: 4 optical bands + 1 numerical bathymetry channel.
   * **8CH**: Complete 8-band SuperDove optical imagery.
   * **9CH**: 8 optical bands + 1 numerical bathymetry channel.
3. **Decoupled Feature Fusion Strategies (Late Fusion vs. Early Fusion)**:
   * Employs `DepthFeatureExtractor` to downsample bathymetric inputs by a factor of 16 through strided convolutions, concatenating topographic embeddings directly at the bottleneck.
   * Controllable via the `use_late_fusion` flag: setting it to `True` enables decoupled late fusion (protecting shallow optical attention from dominant bathymetric gradients), while `False` reverts to early fusion (channels concatenated at the input layer).
4. **Automated Checkpoint Inspection**:
   * The inference pipeline automatically determines the expected channel count, fusion mode, and structural dimensions from saved `.pth` weight files, removing the need for manual configuration edits during deployment.
5. **Spatial Block Partitioning (Preventing Data Leakage)**:
   * Enforces a $1024 \times 1024\text{ m}$ checkerboard spatial block split into an independent validation holdout (15%) and training pool (85%), preventing autocorrelation leakage across overlapping sliding patches.

---

## System Architecture & Workflow

```
[Local Workstation]
  0_local_preprocess.py (Preprocessing Entrypoint)
    ├── Call src.utils.build_spatial_grid_from_roi (Defines EPSG:32650 & 3.0 m affine grid)
    ├── Call src.utils.prepare_feature_stack (Extracts 4CH/8CH, optional bathymetry stacking)
    ├── Filter valid optical/water mask (0 < depth <= 10.0 m)
    ├── Extract sliding patches (256x256, stride=128, 50% overlap)
    └── Spatial checkerboard split (1024 m blocks)
         └── Export: preprocessed_benthic_data_XCH.npz (Includes channel metadata)
                     │
                     ▼ Upload to Google Drive / Cloud Storage
[Cloud Environment (Google Colab / GPU Node)]
  1_colab_train.py
    ├── Load cached .npz tensors
    ├── Instantiate BHUNet_LateFusion (Auto-configures op_channels, depth_in_channels)
    ├── Optimize CombinedLoss: 0.6 * Weighted-CE + 0.4 * Soft-Dice
    ├── Accelerate via AMP FP16 Mixed Precision + CosineAnnealingLR scheduling
    └── Export optimal checkpoint: best_bh_unet_model_XCH.pth
                     │
                     ▼ Download checkpoint to local workspace
[Local Workstation]
  2_predict_and_evaluate.py (Inference & Evaluation)
    ├── Inspect .pth architecture (Auto-detects channel mode and Late Fusion flag)
    ├── Call shared feature preparation (Guarantees numerical isomorphism)
    ├── Execute full-scene sliding inference with 2D Hanning window smoothing
    ├── Export georeferenced GeoTIFF (EPSG:32650 projection, LZW compressed)
    └── Generate dual validation reports: Spatial Hold-Out vs. Full-Scene Metrics
```

---

## Model Architecture: BHUNet_LateFusion

The core network is defined in `src/model.py`, supporting optical-only, Early Fusion, and Late Fusion operational modes:

```
[Optical Input (4CH or 8CH)] ──► Enc1 (16) ──► Enc2 (32) ──► Enc3 (64) ──► Enc4 (128) ──► Bottleneck (256)
                                                                                              │
[Bathymetry Input (1CH)]     ──► DepthFeatureExtractor (4x Stride=2 Convs, outputs 32ch) ───┘
                                                                                              │ Concat
                                                                                    [Fusion Layer: 288 -> 256]
                                                                                              │
[Output (6 Benthic Classes)] ◄── FinalConv ◄── Dec1 (16) ◄── Dec2 (32) ◄── Dec3 (64) ◄── Dec4 (128)
```

* **Channel-wise Attention**: Dynamically calibrates feature responses across spectral channels in encoder and decoder blocks to emphasize water-penetrating wavelengths.
* **Group Normalization (GN)**: Replaces Batch Normalization across all convolutional blocks to ensure gradient stability when training with small batch sizes (e.g., $N = 4 \sim 8$).
* **DepthFeatureExtractor**: Decouples smooth bathymetric gradients into macro-topographic context vectors ($1/16$ scale), preventing bathymetric features from interfering with fine optical textures in early layers.

---

## Data Specifications, Band Configs & Outputs

### 1. Allen Coral Atlas (ACA) Standard 6 Benthic Classes
* **[1] Seagrass**: Seagrass meadows and beds
* **[2] Coral/Algae**: Live hermatypic corals and macroalgal assemblages
* **[3] Sand**: Unconsolidated carbonate/siliciclastic sands
* **[4] Microalgal Mats**: Benthic microalgae / microbial film
* **[5] Rock**: Hardground, pavement, and consolidated limestone substrate
* **[6] Rubble**: Unconsolidated coral skeleton rubble and gravel

### 2. Satellite Imagery & Bathymetric Layers
* **PlanetScope SuperDove Multispectral Imagery (8 Bands)**:
  * Bands: `Coastal Blue (1)`, `Blue (2)`, `Green-I (3)`, `Green (4)`, `Yellow (5)`, `Red (6)`, `Red Edge (7)`, `NIR (8)`.
  * In **4CH** mode, the system extracts indices **`[2, 4, 6, 8]`** (`Blue`, `Green`, `Red`, `NIR`).
* **Bathymetric Digital Elevation Model (DEM)**:
  * High-resolution shallow-water bathymetry ($0 \sim 10.0\text{ m}$), resampled to 3.0 m spatial resolution using bicubic interpolation.

### 3. Generated Artifacts
* `preprocessed_benthic_data_XCH.npz`: Spatial block-partitioned tensor archive containing patch arrays, validation split masks, and channel metadata.
* `best_bh_unet_model_XCH_*.pth`: Optimized model weights containing architecture metadata.
* `habitat_map_pred.tif`: Classified benthic habitat GeoTIFF with projection metadata (`EPSG:32650`, LZW compression).
* `evaluation_report.log`: Comprehensive evaluation logs containing confusion matrices, Overall Accuracy (OA), Cohen's Kappa, Producer's Accuracy (PA), User's Accuracy (UA), Intersection over Union (IoU), and Macro F1 scores.

### Visualization & Evaluation Artifacts

#### 1. Predicted Benthic Habitat Map
![benthic_habitat_comparison_gt_vs_pred](https://github.com/user-attachments/assets/29fdafa4-95f4-4dff-9a21-a983284923d2)
*Figure 1: Full-scene benthic habitat prediction versus Ground Truth reference.*

#### 2. Confusion Matrix
![confusion_matrix_strict_spatial_mean_20runs](https://github.com/user-attachments/assets/ab9bcdc0-1988-44df-ac0e-a51df1e7b36a)
*Figure 2: Mean confusion matrix evaluated over 20 repeated spatial hold-out runs.*

#### 3. Metric Distribution Boxplots
![evaluation_strict_spatial_boxplots](https://github.com/user-attachments/assets/15bacdd3-e1d7-4402-a64d-fea115a072e6)
*Figure 3: Metric distribution boxplots (PA, UA, F1, OA, Kappa) across independent runs.*

#### 4. Spatial Dataset Distribution
![npz_spatial_and_class_distribution_geo](https://github.com/user-attachments/assets/13aff47b-ec54-463b-b70c-17a87bd9be1e)
*Figure 4: Spatial and categorical geographic distribution of extracted training and validation patches.*

---

## Repository Structure

```
Dongsha_benthic_habitat_unet/
├── data/                                 # Geographic input files (git-ignored)
│   ├── PlanetScope_20200518_composite.tif
│   ├── fine_grid_5m_waterdepth.tif
│   ├── mask_aligned.tif
│   └── roi_10m.geojson
├── output/                               # Generated artifacts and logs
│   ├── preprocessed_benthic_data_9CH.npz
│   ├── best_bh_unet_model_9CH_LF.pth
│   ├── habitat_map_pred.tif
│   └── evaluation_report.log
├── notebooks/
│   └── 1_colab_train.ipynb               # Cloud training notebook
├── src/                                  # Modular library components
│   ├── __init__.py                       # Package exports
│   ├── model.py                          # BHUNet_LateFusion, DepthFeatureExtractor, ConvBlock
│   ├── loss.py                           # CombinedLoss (Weighted Cross-Entropy + Soft Dice)
│   └── utils.py                          # Feature stack preparation, alignment, inference, evaluation
├── 0_local_preprocess.py                 # Local preprocessing entrypoint
├── 1_colab_train.py                      # Training execution script
├── 2_predict_and_evaluate.py             # Inference, mapping, and evaluation script
├── environment.yml                       # Conda environment definition
├── requirements.txt                      # Python dependencies list
├── .gitignore
└── README.md                             # Project documentation
```

---

## Key Configurable Parameters

| Script / Module | Parameter | Default | Description & Guidelines |
| :--- | :--- | :---: | :--- |
| `0_local_preprocess.py` | `include_depth` | `True` | Appends bathymetry as an additional channel (`False` for optical-only). |
| | `target_band_indices` | `None` | Band selection list. `None` loads all bands; `[2, 4, 6, 8]` extracts RGB + NIR. |
| | `target_res` | `3.0` | Target spatial resolution in meters. |
| | `z_max` | `10.0` | Maximum depth cutoff in meters for optical penetrability. |
| | `patch_size` / `stride` | `256` / `128` | Patch dimensions and sliding stride (`stride=128` provides 50% overlap). |
| `1_colab_train.py` | `flag_use_late_fusion`| `True` | Enables the decoupled Late Fusion branch for bathymetry. |
| | `batch_size` | `32` | Training batch size (recommended: 32 on standard Colab T4 GPUs). |
| | `lr` | `0.001` | Initial AdamW learning rate with Cosine Annealing decay. |
| | `epochs` / `early_stopping_patience` | `80` / `15` | Total epochs and early stopping tolerance. |
| `2_predict_and_evaluate.py` | `batch_size` | `8` | Inference batch size (e.g., 8 on 4 GB VRAM devices). |
| | `stride` | `128` | Sliding inference stride paired with a 2D Hanning window to eliminate seams. |

---

## Execution Guide (Local & Cloud)

### 1. Data Preprocessing (`0_local_preprocess.py`)
Configure channel options and paths in `0_local_preprocess.py`, then run:
```bash
python 0_local_preprocess.py
```
Upload the resulting `.npz` archive to your cloud storage environment.

### 2. Model Training (`1_colab_train.py` or Notebook)
Mount cloud storage and initiate training:
```python
from src.utils import run_training

# Example: Run 9CH Late Fusion training
run_training(
    data_path="./preprocessed_benthic_data_9CH.npz",
    output_path="./output/best_bh_unet_model_9CH_LF.pth",
    epochs=80,
    batch_size=32,
    lr=0.001,
    early_stopping_patience=15,
    flag_use_late_fusion=True,
    depth_in_channels=1,
    depth_channels=32
)
```
Download the resulting checkpoint (`.pth`) to the local `output/` directory upon completion.

### 3. Inference and Spatial Evaluation (`2_predict_and_evaluate.py`)
Run the standalone evaluation pipeline with automated checkpoint inspection:
```bash
python 2_predict_and_evaluate.py
```
The script outputs a georeferenced GeoTIFF classification map along with independent hold-out and full-scene confusion matrices.

---

## Experimental Results & Benchmark Analysis

Evaluated over the same geographical study area (shallow waters of Dongsha Atoll), comparative performance across input configurations is summarized below:

| Model Architecture | Input Channels | Fusion Strategy | Full-Scene OA | Full-Scene Kappa | Spatial Hold-Out OA | Performance Summary |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **4CH Optical** | RGB + NIR (4) | None | 67.59% | 0.5644 | 64.76% | Baseline optical model; lowest computational demand. |
| **5CH Early Fusion** | RGB + NIR + Depth (5) | Early Conv | ~66.5% | ~0.5500 | ~66.0% | Smooth bathymetric signals introduce slight interference in early layers. |
| **5CH Late Fusion** | RGB + NIR + Depth (5) | Late Fusion | **66.98%** | **0.5623** | **67.28%** | Decoupling preserves optical features; Seagrass PA reaches 79.2%. |
| **8CH Optical** | SuperDove All Bands (8) | None | 69.57% | 0.5927 | 68.44% | Additional spectral bands significantly improve overall class separability. |
| **9CH Early Fusion** | 8 Bands + Depth (9) | Early Conv | 67.12% | 0.5556 | 66.85% | Early fusion is susceptible to premature convergence driven by depth gradients. |
| **9CH Late Fusion** | 8 Bands + Depth (9) | Late Fusion | **72.02%** | **0.6261** | **68.44%** | **Highest overall accuracy**; marked improvements in Seagrass and Microalgae. |
