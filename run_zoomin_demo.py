#!/usr/bin/env python3
"""
Cascaded Zoom-in Detector 簡化重現示範：
在空拍圖上先做全圖偵測，再對高分數區域裁切放大後二次偵測（對應論文 zoom-in 概念）。
使用 torchvision Faster R-CNN（COCO 預訓練）作為基礎偵測器。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT))
from device_utils import choose_device  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "sample_images"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SCORE_TH = 0.5
ZOOM_SIZE = 640


def load_model(device: torch.device):
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device).eval()
    return model, weights.meta["categories"]


def detect(model, image: Image.Image, device: torch.device):
    t = F.to_tensor(image).to(device)
    with torch.no_grad():
        pred = model([t])[0]
    boxes = pred["boxes"].cpu().tolist()
    scores = pred["scores"].cpu().tolist()
    labels = pred["labels"].cpu().tolist()
    keep = [i for i, s in enumerate(scores) if s >= SCORE_TH]
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]


def zoom_crops(image: Image.Image, boxes, margin=0.2):
    w, h = image.size
    crops = []
    meta = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 - margin * bw)
        y1 = max(0, y1 - margin * bh)
        x2 = min(w, x2 + margin * bw)
        y2 = min(h, y2 + margin * bh)
        crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
        if crop.width < 8 or crop.height < 8:
            continue
        crop = crop.resize((ZOOM_SIZE, ZOOM_SIZE))
        crops.append(crop)
        meta.append({"crop_id": i, "orig_box": box})
    return crops, meta


def main():
    use_cuda, msg = choose_device()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"[Device] {msg}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 使用 SIFDAL 的 State-Air 樣本當空拍測試圖
    src = REPRO_ROOT / "sifdal" / "State-Air" / "JPEGImages"
    if src.exists():
        for jpg in list(src.glob("*.jpg"))[:5]:
            dst = DATA_DIR / jpg.name
            if not dst.exists():
                dst.symlink_to(jpg.resolve())

    images = sorted(DATA_DIR.glob("*.jpg"))
    if not images:
        raise SystemExit("找不到 sample_images，請先執行 sifdal 的 setup 或放入 jpg")

    model, categories = load_model(device)
    summary = []

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        boxes1, scores1, labels1 = detect(model, image, device)
        crops, crop_meta = zoom_crops(image, boxes1)
        zoom_detections = 0
        for crop, cm in zip(crops, crop_meta):
            b2, s2, l2 = detect(model, crop, device)
            zoom_detections += len(b2)
            cm["zoom_detections"] = len(b2)

        rec = {
            "image": img_path.name,
            "full_image_detections": len(boxes1),
            "zoom_pass_detections": zoom_detections,
            "crops": len(crops),
        }
        summary.append(rec)
        print(rec)

    result = {
        "method": "Faster R-CNN + cascaded zoom-in (simplified)",
        "device": str(device),
        "num_images": len(summary),
        "details": summary,
        "note": "課程簡化重現，驗證兩階段 zoom-in 流程可執行；非論文完整 Detectron2 訓練結果。",
    }
    out_json = OUT_DIR / "zoomin_demo_results.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
