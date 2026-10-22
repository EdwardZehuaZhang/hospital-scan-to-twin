"""Short fine-tune of a Pointcept (Sonata) model on our hospital labels: the PTv3 backbone stays frozen and a small
MLP head is trained on its per-point features to label points as background or one of six object classes, leave one
area out (train on two areas, test on the third). Test-area points are then clustered into boxes.

python finetune_head.py <bench_dir> <gt_dir>     writes pred_sonata_ft/<area>.json and finetune_log.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from matplotlib.path import Path

sys.path.insert(0, "/mnt/e/WSL")
B, G = sys.argv[1], sys.argv[2]
AREAS = ["lobby", "corridor", "campus"]
CLASSES = ["background", "seating", "table", "kiosk", "sign", "plant", "other"]
BINARY = os.environ.get("BINARY") == "1"  # object vs background only (class sets differ between areas)
OUT = "pred_sonata_ft_binary" if BINARY else "pred_sonata_ft"
torch.manual_seed(0); np.random.seed(0)


def local_box_polys(boxes, meta):
    th = meta["theta"]; c, s = np.cos(th), np.sin(th); R = np.array([[c, -s], [s, c]]); ctr = np.array(meta["centre"])
    out = []
    for b in boxes:
        Rb = np.array([[np.cos(b["yaw"]), -np.sin(b["yaw"])], [np.sin(b["yaw"]), np.cos(b["yaw"])]])
        cw = (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [b["sx"] / 2 + 0.03, b["sy"] / 2 + 0.03]) @ Rb.T + [b["cx"], b["cy"]]
        cl = (cw - ctr) @ R  # world -> local
        out.append((cl, b["z0"] - 0.05, b["z0"] + b["sz"] + 0.05, CLASSES.index(b["cls"])))
    return out


data = {}
for a in AREAS:
    d = np.load(os.path.join(B, f"{a}_sonata_feat.npz")); meta = json.load(open(os.path.join(B, f"{a}_meta.json")))
    gt = json.load(open(os.path.join(G, a, "gt_boxes.json")))
    X = d["xyz"]; y = np.zeros(len(X), np.int64)
    for poly, z0, z1, k in local_box_polys([b for b in gt if b["n_pts"] >= 80], meta):
        m = Path(poly).contains_points(X[:, :2]) & (X[:, 2] >= z0) & (X[:, 2] <= z1)
        y[m] = 1 if BINARY else k
    data[a] = (torch.tensor(d["feat"].astype(np.float32)), torch.tensor(y), X, meta)
    print(a, len(X), {CLASSES[k]: int((y == k).sum()) for k in range(len(CLASSES))}, flush=True)

from bench_boxes import boxes_from_labels, to_world  # noqa: E402

os.makedirs(os.path.join(B, OUT), exist_ok=True)
log = {}
for test in AREAS:
    train = [a for a in AREAS if a != test]
    Xtr = torch.cat([data[a][0] for a in train]); ytr = torch.cat([data[a][1] for a in train])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    cnt = torch.bincount(ytr, minlength=2 if BINARY else len(CLASSES)).float()
    w = torch.where(cnt > 0, (cnt.sum() / (cnt + 1)) ** 0.5, torch.zeros_like(cnt)); w = w / w[w > 0].mean()
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2 if BINARY else len(CLASSES))).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss(weight=w.cuda())
    Xg, yg = ((Xtr - mu) / sd).cuda(), ytr.cuda()
    for ep in range(25):
        perm = torch.randperm(len(Xg), device="cuda"); tot = 0
        net.train()
        for i in range(0, len(perm), 16384):
            j = perm[i:i + 16384]
            loss = lossf(net(Xg[j]), yg[j]); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(j)
    net.eval()
    Xt, yt, Xyz, meta = data[test]
    with torch.no_grad():
        pred = net(((Xt - mu) / sd).cuda()).argmax(-1).cpu().numpy()
    yt = yt.numpy()
    obj_t, obj_p = yt > 0, pred > 0
    iou_obj = float((obj_t & obj_p).sum() / max(1, (obj_t | obj_p).sum()))
    names = ["background", "other"] if BINARY else CLASSES
    mapping = {c: c for c in names[1:]}
    bx = to_world(boxes_from_labels(Xyz, pred, names, mapping, min_pts=25), meta)
    json.dump(bx, open(os.path.join(B, OUT, f"{test}.json"), "w"), indent=1)
    log[test] = {"train": train, "final_loss": tot / len(perm), "object_point_iou": iou_obj, "boxes": len(bx),
                 "point_acc_on_objects": float((pred[obj_t] == yt[obj_t]).mean()) if obj_t.any() else None}
    print(test, log[test], flush=True)
json.dump(log, open(os.path.join(B, OUT + "_log.json"), "w"), indent=1)
