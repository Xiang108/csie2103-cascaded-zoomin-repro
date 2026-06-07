#!/usr/bin/env python3
"""VisDrone 原生標註 → COCO JSON + 影像 symlink（RemDet / Detectron2 用）。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "datasets"))
DET = DATA_ROOT / "VisDrone-DET"
COCO = DATA_ROOT / "VisDrone2019-DET-COCO"

CATEGORIES = [
    {"id": 1, "name": "pedestrian", "supercategory": "object"},
    {"id": 2, "name": "people", "supercategory": "object"},
    {"id": 3, "name": "bicycle", "supercategory": "object"},
    {"id": 4, "name": "car", "supercategory": "object"},
    {"id": 5, "name": "van", "supercategory": "object"},
    {"id": 6, "name": "truck", "supercategory": "object"},
    {"id": 7, "name": "tricycle", "supercategory": "object"},
    {"id": 8, "name": "awning-tricycle", "supercategory": "object"},
    {"id": 9, "name": "bus", "supercategory": "object"},
    {"id": 10, "name": "motor", "supercategory": "object"},
]

SPLITS = [
    ("train", "VisDrone2019-DET-train", "VisDrone2019-DET_train_coco.json"),
    ("val", "VisDrone2019-DET-val", "VisDrone2019-DET_val_coco.json"),
]


def link_images(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for jpg in src_dir.glob("*.jpg"):
        link = dst_dir / jpg.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(jpg.resolve())


def convert_split(split: str, folder: str, out_name: str, image_id_start: int, ann_id_start: int):
    src = DET / folder
    img_dir = src / "images"
    ann_dir = src / "annotations"
    if not img_dir.is_dir():
        raise SystemExit(f"找不到 {img_dir}，請先執行 download_visdrone.py")

    link_images(img_dir, COCO / "images" / split)

    images, annotations = [], []
    image_id = image_id_start
    ann_id = ann_id_start

    for ann_path in sorted(ann_dir.glob("*.txt")):
        jpg = img_dir / ann_path.with_suffix(".jpg").name
        if not jpg.exists():
            continue
        from PIL import Image

        with Image.open(jpg) as im:
            w, h = im.size

        images.append({"id": image_id, "file_name": jpg.name, "width": w, "height": h})
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, bw, bh = map(float, parts[:4])
            score = int(parts[4])
            cat = int(parts[5])
            if score == 0 or cat <= 0 or cat > 10:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        image_id += 1

    out = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    (COCO / "annotations").mkdir(parents=True, exist_ok=True)
    out_path = COCO / "annotations" / out_name
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"{split}: {len(images)} images, {len(annotations)} anns -> {out_path.name}")
    return image_id, ann_id


def is_ready() -> bool:
    train_json = COCO / "annotations" / "VisDrone2019-DET_train_coco.json"
    val_json = COCO / "annotations" / "VisDrone2019-DET_val_coco.json"
    return train_json.exists() and val_json.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.force and is_ready():
        print(f"[OK] COCO 已存在：{COCO}")
        return

    COCO.mkdir(parents=True, exist_ok=True)
    next_img, next_ann = 1, 1
    for split, folder, out_name in SPLITS:
        next_img, next_ann = convert_split(split, folder, out_name, next_img, next_ann)
    print(f"[OK] {COCO}")


if __name__ == "__main__":
    main()
