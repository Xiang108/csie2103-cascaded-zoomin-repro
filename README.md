# CSIE2103 重現紀錄：Cascaded Zoom-in Detector

學號：**M11417015**　姓名：**謝宇翔**  
課程：CSIE 2103 類神經網路｜Spring 2026

## 論文

**Cascaded Zoom-in Detector for High Resolution Aerial Images**（CVPR 2023 EarthVision）  
官方程式：https://github.com/akhilpm/DroneDetectron2

## 本 repo 內容

本 repo 為**期末作業重現結果紀錄**，包含：

| 檔案 | 說明 |
|------|------|
| [`results.json`](results.json) | 重現數值摘要 |
| [`figures/`](figures/) | 訓練曲線、AP 對照圖 |
| [`outputs/visdrone_train/`](outputs/visdrone_train/) | 訓練 metrics 與設定摘要 |

> 模型權重（`.pth`）體積過大，未上傳至此 repo。

## 重現設定

| 項目 | 設定 |
|------|------|
| 框架 | Detectron2 + DroneDetectron2（RCNN-FPN-CROP） |
| 資料集 | VisDrone2019-DET（train 6471 / val 548） |
| 訓練 | 90000 iter |
| 環境 | Python 3.12、PyTorch 2.12、detectron2 0.6、RTX 5090 |

## 重現結果（VisDrone validation）

| 指標 | 本次重現 |
|------|----------|
| AP | **33.4%** |
| AP50 | **58.1%** |

## 如何重現（參考官方 repo）

```bash
git clone https://github.com/akhilpm/DroneDetectron2.git
cd DroneDetectron2
# 依官方 README 安裝 Detectron2、準備 VisDrone 並訓練 RCNN-FPN-CROP
```

本 repo 的 `results.json` 與 `figures/` 可作為對照基準。

## 差距原因與改善方向

- **可能原因**：backbone / 預訓練權重與論文表格設定略有差異
- **改善方向**：對齊論文預訓練權重與完整 eval 流程
