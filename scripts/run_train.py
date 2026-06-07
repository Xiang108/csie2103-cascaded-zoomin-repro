#!/usr/bin/env python3
"""Cascaded Zoom-in Detectron2 完整 VisDrone 訓練（支援 --resume）。"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZOOMIN = ROOT / "cascaded-zoomin"
REPRO = ROOT
DATA = Path(os.environ.get("ZOOMIN_DATA_DIR", str(ROOT / "datasets" / "VisDrone-Zoomin")))
LOG = REPRO / "outputs" / "train.log"
RESULT = REPRO / "results.json"
OUT_DIR = REPRO / "outputs" / "visdrone_train"


def parse_metrics(log_text: str) -> dict:
    metrics = {}
    # detectron2 table: | 18.474 | 35.992 | ...
    for line in reversed(log_text.splitlines()):
        if re.match(r"^\|\s*[\d.]+\s*\|", line.strip()):
            parts = [p.strip() for p in line.strip().strip("|").split("|") if p.strip()]
            if len(parts) >= 2:
                metrics["bbox_AP"] = float(parts[0]) / 100.0
                metrics["bbox_AP50"] = float(parts[1]) / 100.0
            break
    for pat, key in [
        (r"bbox/AP:\s*([\d.]+)", "bbox_AP"),
        (r"AP:\s*([\d.]+)", "bbox_AP"),
        (r"AP50:\s*([\d.]+)", "bbox_AP50"),
    ]:
        hits = re.findall(pat, log_text)
        if hits:
            metrics[key] = float(hits[-1])
    return metrics


def main():
    REPRO.mkdir(parents=True, exist_ok=True)
    (REPRO / "outputs").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.check_call([sys.executable, str(ROOT / "scripts/prepare_data.py")])

    gpu = os.environ.get("ZOOMIN_GPU", "1")
    max_iter = int(os.environ.get("ZOOMIN_MAX_ITER", "90000"))
    resume = OUT_DIR.joinpath("last_checkpoint").exists() or any(OUT_DIR.glob("model_*.pth"))
    if os.environ.get("ZOOMIN_FRESH", "") == "1":
        resume = False

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["VISDRONE_DATA_DIR"] = str(DATA.resolve())
    env["PYTHONPATH"] = str(ZOOMIN.resolve()) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cfg = ZOOMIN / "configs" / "RCNN-FPN-CROP.yaml"
    cmd = [
        sys.executable, "train_net.py",
        "--num-gpus", "1",
        "--config-file", str(cfg),
    ]
    if resume:
        cmd.append("--resume")
    cmd.extend([
        "OUTPUT_DIR", str(OUT_DIR.resolve()),
        "SOLVER.MAX_ITER", str(max_iter),
        "TEST.EVAL_PERIOD", "3000",
        "SOLVER.CHECKPOINT_PERIOD", "3000",
    ])
    if not resume:
        cmd.extend(["MODEL.WEIGHTS", "detectron2://ImageNetPretrained/MSRA/R-50.pkl"])

    if resume:
        print(f"[Zoom-in] 從 checkpoint 續訓（{OUT_DIR}）")
    print(f"[Zoom-in] GPU={gpu} MAX_ITER={max_iter} resume={resume}")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n\n===== restart {datetime.now().isoformat()} resume={resume} =====\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=ZOOMIN, env=env, stdout=f, stderr=subprocess.STDOUT)

    log_text = LOG.read_text(encoding="utf-8", errors="ignore")
    # 只檢查最近一次 restart 之後的 log
    if "===== restart" in log_text:
        log_text_tail = log_text.split("===== restart")[-1]
    else:
        log_text_tail = log_text[-8000:]
    metrics = parse_metrics(log_text)
    ok = proc.returncode == 0 and "Traceback" not in log_text_tail
    last_iter = 0
    hits = re.findall(r"iter:\s*(\d+)", log_text_tail)
    if hits:
        last_iter = int(hits[-1])

    result = {
        "paper": "Cascaded Zoom-in Detector (CVPR 2023 EarthVision)",
        "dataset": "VisDrone2019-DET full train 6471 + val 548",
        "training": True,
        "max_iter": max_iter,
        "last_iter": last_iter,
        "device": f"cuda:{gpu}",
        "output_dir": str(OUT_DIR.relative_to(ROOT.parent)),
        "log": str(LOG.relative_to(ROOT.parent)),
        "metrics": metrics,
        "status": "completed" if ok and last_iter >= max_iter - 1 else ("failed" if not ok else "training"),
    }
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if ok and last_iter >= max_iter - 1 else (1 if not ok else 2))


if __name__ == "__main__":
    main()
