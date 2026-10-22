"""Boxes from our image-based twin pipeline (v3): the oriented box of each generated object's LiDAR points, with its
automatic label mapped to the benchmark classes. python pipeline_boxes.py <twin_dir> <out_dir>"""
import csv, json, os, sys
import numpy as np
W, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
MAP = {"chair": "seating", "bench": "seating", "chair bench": "seating", "table": "table", "bench table": "table",
       "kiosk machine": "kiosk", "vending machine": "kiosk", "kiosk machine vending machine": "kiosk",
       "standing sign board": "sign", "wet floor sign": "sign", "display stand": "sign", "queue stanchion": "other",
       "potted plant": "plant", "trash bin": "other"}
for area in ("lobby", "corridor", "campus"):
    g = os.path.join(W, area, "gen3"); bx = []
    for r in csv.DictReader(open(os.path.join(g, "metrics.csv"))):
        p = os.path.join(g, f"{r['id']}_pts.npy")
        if not os.path.exists(p):
            continue
        Y = np.load(p)
        xy = Y[:, :2] - Y[:, :2].mean(0); w, v = np.linalg.eigh(np.cov(xy.T)); th = float(np.arctan2(v[1, 1], v[0, 1]))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); q = xy @ R
        lo, hi = np.percentile(q, 2, 0), np.percentile(q, 98, 0); c = Y[:, :2].mean(0) + R @ ((lo + hi) / 2)
        bx.append({"cx": float(c[0]), "cy": float(c[1]), "sx": float(hi[0] - lo[0]), "sy": float(hi[1] - lo[1]), "yaw": th,
                   "z0": float(np.percentile(Y[:, 2], 2)), "sz": float(np.ptp(np.percentile(Y[:, 2], [2, 98]))),
                   "cls": MAP.get(r["label"], "other"), "score": float(r["views"]), "id": r["id"]})
    json.dump(bx, open(os.path.join(out, f"{area}.json"), "w"), indent=1); print(area, len(bx))
