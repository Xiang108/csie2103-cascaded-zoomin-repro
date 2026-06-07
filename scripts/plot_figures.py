#!/usr/bin/env python3
"""Cascaded Zoom-in 重現：從 train.log 產生報告用圖表。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT
OUT = REPRO / "figures"
LOG = REPRO / "outputs" / "train.log"
RESULT = REPRO / "results.json"

PAPER_AP = 35.0
PAPER_AP50 = 55.0


def parse_log() -> tuple[list[int], list[float], list[float], list[float]]:
    text = LOG.read_text(encoding="utf-8", errors="ignore") if LOG.exists() else ""
    iters, losses, aps, aps50 = [], [], [], []
    recent = text.split("===== restart")[-1]
    for m in re.finditer(
        r"iter:\s*(\d+).*?total_loss:\s*([\d.]+)", recent
    ):
        iters.append(int(m.group(1)))
        losses.append(float(m.group(2)))
    for line in recent.splitlines():
        if "copypaste:" in line and re.search(r"copypaste: [\d]", line):
            vals = line.split("copypaste:")[-1].strip().split(",")
            if len(vals) >= 2:
                step = len(aps) * 3000 + 3000
                aps.append(float(vals[0]))
                aps50.append(float(vals[1]))
    return iters, losses, aps, aps50


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    iters, losses, aps, aps50 = parse_log()
    result = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    metrics = result.get("metrics", {})
    our_ap = float(metrics.get("bbox_AP", aps[-1] / 100 if aps else 0)) * (
        1 if float(metrics.get("bbox_AP", 0)) <= 1 else 0.01
    )
    our_ap50 = float(metrics.get("bbox_AP50", aps50[-1] / 100 if aps50 else 0)) * (
        1 if float(metrics.get("bbox_AP50", 0)) <= 1 else 0.01
    )
    if our_ap > 1:
        our_ap /= 100
    if our_ap50 > 1:
        our_ap50 /= 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if iters:
        axes[0].plot(iters, losses, color="#2563eb", lw=1.2)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Total loss")
        axes[0].set_title("Training loss (90000 iter)")
        axes[0].grid(alpha=0.3)
    if aps:
        xs = list(range(3000, 3000 * len(aps) + 1, 3000))
        axes[1].plot(xs, aps, "o-", label="AP", color="#2563eb")
        axes[1].plot(xs, aps50, "s-", label="AP50", color="#16a34a")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("AP (%)")
        axes[1].set_title("Validation AP (eval every 3000 iter)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "01_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Paper (approx)", "Ours"]
    vals = [PAPER_AP, our_ap * 100]
    colors = ["#16a34a", "#2563eb"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("bbox AP (%)")
    ax.set_title("VisDrone val AP comparison")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "02_ap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "figures": ["01_training_curves.png", "02_ap_comparison.png"],
        "metrics": {"bbox_AP": our_ap, "bbox_AP50": our_ap50},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
