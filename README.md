# CSIE2103 重現：Cascaded Zoom-in Detector

學號：**M11417015**　姓名：**謝宇翔**  
課程：CSIE 2103 類神經網路｜Spring 2026

## 論文

**Cascaded Zoom-in Detector for High Resolution Aerial Images**（CVPR 2023 EarthVision）  
官方程式：https://github.com/akhilpm/DroneDetectron2

## 我們的重現結果（VisDrone validation）

| 指標 | 本次重現 |
|------|----------|
| AP | **33.4%** |
| AP50 | **58.1%** |
| 訓練 | Detectron2 RCNN-FPN-CROP，**90000 iter** |
| 資料集 | train 6471 / val 548 | 

詳見 [`results.json`](results.json)、[`figures/`](figures/)。

## 如何重現

```bash
cd reproduction
python3 scripts/setup_zoomin_dataset.py
python3 scripts/run_zoomin_full_train.py
python3 scripts/plot_zoomin_figures.py
```

環境：Python 3.12、PyTorch 2.12、detectron2 0.6、RTX 5090。

訓練輸出目錄（本機）：`reproduction/csie2103-cascaded-zoomin-repro/outputs/visdrone_train/`  
（模型權重 `.pth` 太大，未上傳 GitHub，請依上方腳本自行訓練。）

## 與論文差距

- AP 可能略低於部分論文表格（backbone / 預訓練差異）
- 已跑滿 90000 iter，資料集完整

## 改善方向

- 對齊論文預訓練權重與完整 eval 流程
