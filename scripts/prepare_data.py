#!/usr/bin/env python3
import json, os, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
base = Path(os.environ.get("DATA_ROOT", ROOT / "datasets"))
DET = base / "VisDrone-DET"
COCO = base / "VisDrone2019-DET-COCO"
OUT = base / "VisDrone-Zoomin"

def link_images(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for jpg in src_dir.glob("*.jpg"):
        link = dst_dir / jpg.name
        if not link.exists():
            link.symlink_to(jpg.resolve())

OUT.mkdir(parents=True, exist_ok=True)
train_img = DET / "VisDrone2019-DET-train" / "images"
val_img = DET / "VisDrone2019-DET-val" / "images"
if not train_img.exists():
    train_img = DET / "split" / "train" / "images"
    val_img = DET / "split" / "val" / "images"
link_images(train_img, OUT / "train" / "images")
link_images(val_img, OUT / "val" / "images")
for split, name in [("train", "VisDrone2019-DET_train_coco.json"), ("val", "VisDrone2019-DET_val_coco.json")]:
    src = COCO / "annotations" / name
    dst = OUT / f"annotations_VisDrone_{split}.json"
    if src.exists() and not dst.exists():
        dst.symlink_to(src.resolve())
meta = {"train_images": len(list((OUT / "train" / "images").glob("*.jpg"))),
        "val_images": len(list((OUT / "val" / "images").glob("*.jpg"))), "data_dir": str(OUT)}
(OUT / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
