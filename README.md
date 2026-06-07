# CSIE2103 重現：Cascaded Zoom-in Detector

學號：**M11417015**　姓名：**謝宇翔**  
課程：CSIE 2103 類神經網路｜Spring 2026

**Cascaded Zoom-in Detector**（CVPR 2023 EarthVision）  
官方：https://github.com/akhilpm/DroneDetectron2

## 重現結果（VisDrone val）

| 指標 | 本次重現 |
|------|----------|
| AP | **33.4%** |
| AP50 | **58.1%** |

90000 iter，完整 VisDrone train 6471 / val 548。

> **資料集會自動下載**：首次執行訓練時會下載 VisDrone train+val（壓縮約 **1.5 GB**，解壓後約 **43 GB**），不需手動下載、也不會 push 至 GitHub。詳見 [`datasets/README.md`](datasets/README.md)。


---

## 一鍵訓練

```bash
git clone https://github.com/Xiang108/csie2103-cascaded-zoomin-repro.git
cd csie2103-cascaded-zoomin-repro

python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
pip install pycocotools matplotlib opencv-python-headless pillow

python3 scripts/run_train.py
```

自動流程：**下載 VisDrone → 轉 COCO → 建立 Zoom-in 目錄 → Detectron2 訓練 90000 iter**。

| 環境變數 | 預設 |
|----------|------|
| `ZOOMIN_GPU` | 1 |
| `ZOOMIN_MAX_ITER` | 90000 |
