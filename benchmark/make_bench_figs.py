"""Benchmark figures: (a) F1 per method and area at BEV IoU 0.25; (b) bird's-eye overlays of predictions against the
ground truth for one area. python make_bench_figs.py  -> figures/bench_f1.png, figures/bench_overlay_<area>.png"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import AREAS, in_region, load_pred, rect

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
NAMES = {"lidar_clustering": "LiDAR clustering\n(no learning)", "our_pipeline": "Our image-based\npipeline",
         "sonata_zeroshot": "Sonata\nzero-shot", "sonata_finetuned": "Sonata head\nfine-tuned",
         "slm_base": "SpatialLM\nzero-shot (3 seeds)", "slm_base_prompted_s0": "SpatialLM\ncategory prompt",
         "slm_arkit_cat_s0": "SpatialLM\nARKit-tuned", "slm_base_s0": "SpatialLM zero-shot"}
M = R["methods"]
def f1(m, a):
    return M[m]["all@0.25"]["f1"] if a == "all" else M[m]["areas"].get(a, {}).get("0.25", {}).get("f1", 0)
seeds = [m for m in ("slm_base_s0", "slm_base_s1", "slm_base_s2") if m in M]
order = ["lidar_clustering", "our_pipeline", "sonata_zeroshot", "sonata_finetuned", "slm_base", "slm_base_prompted_s0", "slm_arkit_cat_s0"]
cols = {"lobby": "#c0392b", "corridor": "#e67e22", "campus": "#2471a3", "all": "#333"}
fig, ax = plt.subplots(figsize=(11.5, 3.6))
w = 0.2
for k, a in enumerate(["lobby", "corridor", "campus", "all"]):
    vals, lo, hi = [], [], []
    for m in order:
        if m == "slm_base":
            v = [f1(s, a) for s in seeds]; vals.append(np.mean(v)); lo.append(np.mean(v) - min(v)); hi.append(max(v) - np.mean(v))
        else:
            vals.append(f1(m, a)); lo.append(0); hi.append(0)
    ax.bar(np.arange(len(order)) + (k - 1.5) * w, vals, w, yerr=[lo, hi], capsize=2, color=cols[a], alpha=0.85,
           error_kw={"lw": 0.8}, label={"lobby": "NUH lobby", "corridor": "NUH Renal Centre", "campus": "Campus", "all": "All areas"}[a])
ax.set_xticks(range(len(order))); ax.set_xticklabels([NAMES[m] for m in order], fontsize=8)
ax.set_ylabel("F1 (BEV IoU >= 0.25)"); ax.set_ylim(0, 0.8); ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "bench_f1.png"), dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

for area in ("corridor", "lobby", "campus"):
    meta = json.load(open(os.path.join(HERE, "meta", f"{area}_meta.json")))
    gt = [b for b in json.load(open(os.path.join(HERE, "gt", area, "gt_boxes.json"))) if b["n_pts"] >= 80]
    show = [m for m in ("lidar_clustering", "our_pipeline", "sonata_zeroshot", "slm_base_s0") if m in R["methods"]]
    fig, axs = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.6))
    x0, x1, y0, y1 = AREAS[area]
    for ax, m in zip(np.atleast_1d(axs), show):
        f = glob.glob(os.path.join(HERE, "pred", m, area + ".*"))
        pred = [p for p in load_pred(f[0], meta) if in_region(p, area, meta)] if f else []
        for b in gt:
            c = rect(b); c = np.vstack([c, c[:1]]); ax.fill(c[:, 0], c[:, 1], color="#bbb", alpha=0.6, lw=0)
        for b in pred:
            c = rect(b); c = np.vstack([c, c[:1]]); ax.plot(c[:, 0], c[:, 1], color="#c0392b", lw=0.8)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(NAMES[m].replace("\n", " ") + (" (seed 0)" if m == "slm_base_s0" else "") + f": {len(pred)} boxes", fontsize=8)
    fig.suptitle(f"{area}: ground truth (grey) and predictions (red)", fontsize=9)
    fig.savefig(os.path.join(HERE, "figures", f"bench_overlay_{area}.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
print("figures written")
