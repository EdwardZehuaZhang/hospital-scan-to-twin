"""Sonata (Pointcept PTv3, self-supervised) on the benchmark areas, in tiles: per-point ScanNet-20 semantic labels from the
released linear-probe head (zero-shot for our scenes), and per-point backbone features on a 5 cm subsample for the
head fine-tune. Then furniture points are clustered into boxes.

python sonata_bench.py <bench_dir> <area> [<area> ...]
writes <area>_sonata_feat.npz (xyz_local, feat fp16, scannet label) and pred_sonata_zs/<area>.json (world-frame boxes)
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/opt/work/sonata"); sys.path.insert(0, "/mnt/e/WSL")
import sonata  # noqa: E402
from bench_boxes import boxes_from_labels, to_world  # noqa: E402

B = sys.argv[1]
CLS20 = ("wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter",
         "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture")
MAP = {"chair": "seating", "sofa": "seating", "table": "table", "desk": "table", "counter": "other", "cabinet": "other",
       "bookshelf": "other", "refrigerator": "other", "otherfurniture": "other", "bed": "other"}


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__(); self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


model = sonata.load("sonata", repo_id="facebook/sonata",
                    custom_config=dict(enc_patch_size=[1024] * 5, enable_flash=False)).cuda().eval()
ck = sonata.load("sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True)
head = SegHead(**ck["config"]).cuda().eval(); head.load_state_dict(ck["state_dict"])
transform = sonata.transform.default()


def run_tile(coord, color, normal):
    pt = transform({"coord": coord.copy(), "color": color.copy(), "normal": normal.copy()})
    with torch.inference_mode():
        for k in pt:
            if isinstance(pt[k], torch.Tensor):
                pt[k] = pt[k].cuda(non_blocking=True)
        inv = pt["inverse"]
        p = model(pt)
        while "pooling_parent" in p.keys():
            parent = p.pop("pooling_parent"); inverse = p.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, p.feat[inverse]], dim=-1); p = parent
        f = p.feat
        lab = head(f).argmax(-1)
        return f[inv].half().cpu().numpy(), lab[inv].cpu().numpy()


os.makedirs(os.path.join(B, "pred_sonata_zs"), exist_ok=True)
for area in sys.argv[2:]:
    d = np.load(os.path.join(B, f"{area}_sonata.npz")); meta = json.load(open(os.path.join(B, f"{area}_meta.json")))
    X, C, N = d["coord"], d["color"], d["normal"]
    floor = np.percentile(X[:, 2], 2)
    keep = X[:, 2] < floor + 2.6
    X, C, N = X[keep], C[keep], N[keep]
    # 5 cm subsample whose features are kept
    g5 = np.floor(X / 0.05).astype(np.int64); _, rep = np.unique(g5, axis=0, return_index=True)
    sel = np.zeros(len(X), bool); sel[rep] = True
    lab_all = np.full(len(X), -1, np.int64)
    selidx = np.where(sel)[0]; pos = np.full(len(X), -1, np.int64); pos[selidx] = np.arange(len(selidx))
    F = np.zeros((len(selidx), 1232), np.float16); got = np.zeros(len(selidx), bool)
    core, margin = 4.0, 0.75
    xs = np.arange(X[:, 0].min(), X[:, 0].max(), core); ys = np.arange(X[:, 1].min(), X[:, 1].max(), core)
    for x0 in xs:
        for y0 in ys:
            inner = (X[:, 0] >= x0) & (X[:, 0] < x0 + core) & (X[:, 1] >= y0) & (X[:, 1] < y0 + core)
            if inner.sum() < 500:
                continue
            outer = (X[:, 0] >= x0 - margin) & (X[:, 0] < x0 + core + margin) & (X[:, 1] >= y0 - margin) & (X[:, 1] < y0 + core + margin)
            idx = np.where(outer)[0]
            try:
                f, l = run_tile(X[idx], C[idx], N[idx])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); idx = idx[::2]; f, l = run_tile(X[idx], C[idx], N[idx])
            m = inner[idx]
            lab_all[idx[m]] = l[m]
            ks = m & sel[idx]
            F[pos[idx[ks]]] = f[ks]; got[pos[idx[ks]]] = True
            torch.cuda.empty_cache()
    have = lab_all >= 0
    print(area, "points", len(X), "labelled", int(have.sum()), "features", int(got.sum()), flush=True)
    fi = selidx[got]
    np.savez(os.path.join(B, f"{area}_sonata_feat.npz"), xyz=X[fi].astype(np.float32), feat=F[got], scannet=lab_all[fi],
             floor=np.float32(floor))
    bx = to_world(boxes_from_labels(X[have], lab_all[have], CLS20, MAP), meta)
    json.dump(bx, open(os.path.join(B, "pred_sonata_zs", f"{area}.json"), "w"), indent=1)
    u, c = np.unique(lab_all[have], return_counts=True)
    print(area, {CLS20[k]: int(v) for k, v in zip(u, c)}, "boxes", len(bx), flush=True)
