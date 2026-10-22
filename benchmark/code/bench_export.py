"""Export each benchmark area for the learned models: rotated so the dominant wall direction is along x (SpatialLM
assumes axis-aligned, z-up rooms), cropped to at most 25 m, as a PLY (SpatialLM) and an npz with normals (Sonata).
World coordinates are recovered with: world_xy = R(theta) @ local_xy + centre.

python bench_export.py <bench_dir> <out_dir> <area> [<area> ...]
"""
import json
import os
import sys

import numpy as np
import open3d as o3d

bench, outd = sys.argv[1], sys.argv[2]
os.makedirs(outd, exist_ok=True)


def dominant_angle(XY):
    best = (0, -1)
    for a in np.radians(np.arange(0, 90, 0.5)):
        c, s = np.cos(-a), np.sin(-a)
        q = XY @ np.array([[c, s], [-s, c]])
        hx = np.histogram(q[:, 0], bins=np.arange(q[:, 0].min(), q[:, 0].max() + 0.05, 0.05))[0]
        hy = np.histogram(q[:, 1], bins=np.arange(q[:, 1].min(), q[:, 1].max() + 0.05, 0.05))[0]
        score = float((hx.astype(float) ** 2).sum() + (hy.astype(float) ** 2).sum())
        if score > best[1]:
            best = (a, score)
    return best[0]


for area in sys.argv[3:]:
    d = np.load(os.path.join(bench, area, "area_points.npz"))
    P, C, H = d["P"], d["C"].astype(np.float32), d["H"]
    wall = (H > 1.0) & (H < 2.0)
    sub = np.where(wall)[0][:: max(1, wall.sum() // 400000)]
    th = dominant_angle(P[sub, :2])
    gt = json.load(open(os.path.join(bench, area, "gt_boxes.json")))
    G = np.array([[b["cx"], b["cy"]] for b in gt])
    centre = (G.min(0) + G.max(0)) / 2  # centre the 25 m crop on the annotated objects
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    L = P.copy(); L[:, :2] = (P[:, :2] - centre) @ R  # local = R^T (world - centre)
    keep = np.all(np.abs(L[:, :2]) < 12.5, axis=1)
    L, C = L[keep], C[keep]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(L))
    pc.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    pc = pc.voxel_down_sample(0.02)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    o3d.io.write_point_cloud(os.path.join(outd, f"{area}.ply"), pc, write_ascii=False)
    X = np.asarray(pc.points).astype(np.float32); Cc = (np.asarray(pc.colors) * 255).astype(np.float32)
    Nn = np.asarray(pc.normals).astype(np.float32)
    np.savez_compressed(os.path.join(outd, f"{area}_sonata.npz"), coord=X, color=Cc, normal=Nn)
    json.dump({"theta": float(th), "centre": centre.tolist(), "points": int(len(X)), "cropped_frac": float(1 - keep.mean())},
              open(os.path.join(outd, f"{area}_meta.json"), "w"))
    print(area, "theta deg", round(float(np.degrees(th)), 1), "points", len(X), "cropped", round(float(1 - keep.mean()), 3), flush=True)
