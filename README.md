# CSIE2103 重現：Cascaded Zoom-in Detector

學號：**M11417015**　姓名：**謝宇翔**  
課程：CSIE 2103 類神經網路｜Spring 2026

**Cascaded Zoom-in Detector for High Resolution Aerial Images**（CVPR 2023 EarthVision）  
官方：https://github.com/akhilpm/DroneDetectron2

## 重現結果（VisDrone val）

| 指標 | 本次重現 |
|------|----------|
| AP | **33.4%** |
| AP50 | **58.1%** |

訓練：Detectron2 RCNN-FPN-CROP，90000 iter，完整 VisDrone train 6471 / val 548。

---

## 快速開始

### 1. 環境

```bash
git clone https://github.com/Xiang108/csie2103-cascaded-zoomin-repro.git
cd csie2103-cascaded-zoomin-repro

python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
pip install pycocotools matplotlib opencv-python-headless
```

### 2. 資料集（COCO 格式）

```
datasets/VisDrone2019-DET-COCO/
  annotations/VisDrone2019-DET_train_coco.json
  annotations/VisDrone2019-DET_val_coco.json
  images/train/*.jpg
  images/val/*.jpg
```

同時需要 YOLO 版 VisDrone 供 symlink（見 `datasets/README.md`）。

### 3. 準備 Detectron2 資料目錄

```bash
python3 scripts/prepare_data.py
# 產生 datasets/VisDrone-Zoomin/
```

### 4. 訓練

```bash
python3 scripts/run_train.py
```

| 變數 | 預設 | 說明 |
|------|------|------|
| `ZOOMIN_GPU` | 1 | GPU 編號 |
| `ZOOMIN_MAX_ITER` | 90000 | 訓練 iter |
| `ZOOMIN_FRESH` | — | 設 1 強制從頭訓練 |

### 5. 輸出

| 路徑 | 內容 |
|------|------|
| `outputs/visdrone_train/` | checkpoint、metrics |
| `results.json` | AP / AP50 |
| `figures/` | 訓練曲線、對照圖 |

---

## 目錄結構

```
├── cascaded-zoomin/   # DroneDetectron2 fork
├── scripts/
│   ├── prepare_data.py
│   ├── run_train.py
│   └── plot_figures.py
├── results.json
├── figures/
└── outputs/
```
