# CSIE2103 重現：Cascaded Zoom-in Detector

學號：M11417015　姓名：謝宇翔

## 論文

Cascaded Zoom-in Detector for High Resolution Aerial Images（CVPR 2023 EarthVision / IEEE GRSS）

官方程式：https://github.com/akhilpm/DroneDetectron2

## 本次做法（簡化重現）

因完整 Detectron2 + VisDrone 訓練需大量資料與環境，本次以 **Faster R-CNN 預訓練模型 + zoom-in 兩階段推論** 驗證論文核心流程：

1. 全圖偵測  
2. 對高分數框裁切放大  
3. 在裁切圖上再偵測一次  

## 執行

```bash
cd reproduction/csie2103-cascaded-zoomin-repro
python3 run_zoomin_demo.py
```

結果輸出：`outputs/zoomin_demo_results.json`

## 環境

Python 3.12、PyTorch、torchvision；自動依 GPU 是否空閒選擇裝置。

## 與論文差距

- 未使用論文完整 Detectron2 訓練設定與 VisDrone 官方 mAP 評測。  
- 基礎偵測器為 COCO 預訓練 Faster R-CNN，非論文 FPN-CROP 權重。  
- 屬「流程重現」而非數值完全對齊。

## 改善方向

- 下載 VisDrone 與作者提供的 COCO 格式標註。  
- 依官方 repo 訓練 `RCNN-FPN-CROP.yaml` 後再評估 mAP。
