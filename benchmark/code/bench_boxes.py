"""Shared helpers: cluster labelled points into oriented boxes; local-to-world transform."""
import numpy as np
from sklearn.cluster import DBSCAN


def boxes_from_labels(X, lab, names, mapping, min_pts=60):
    out = []
    for cname in sorted(set(mapping.values())):
        ids = [i for i, n in enumerate(names) if mapping.get(n) == cname]
        m = np.isin(lab, ids)
        if m.sum() < min_pts:
            continue
        Q = X[m]
        g = np.floor(Q / 0.05).astype(np.int64); ug, iq = np.unique(g, axis=0, return_inverse=True)
        cl = DBSCAN(eps=0.11, min_samples=3).fit_predict((ug + 0.5) * 0.05, sample_weight=np.bincount(iq.ravel()))[iq.ravel()]
        for l in set(cl) - {-1}:
            Y = Q[cl == l]
            if len(Y) < min_pts:
                continue
            xy = Y[:, :2] - Y[:, :2].mean(0)
            w, v = np.linalg.eigh(np.cov(xy.T)); th = float(np.arctan2(v[1, 1], v[0, 1]))
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            q = xy @ R; lo, hi = np.percentile(q, 2, 0), np.percentile(q, 98, 0)
            c = Y[:, :2].mean(0) + R @ ((lo + hi) / 2)
            sz = hi - lo
            if max(sz) < 0.2 or max(sz) > 6:
                continue
            out.append({"cx": float(c[0]), "cy": float(c[1]), "sx": float(sz[0]), "sy": float(sz[1]), "yaw": th,
                        "z0": float(np.percentile(Y[:, 2], 2)), "sz": float(np.ptp(np.percentile(Y[:, 2], [2, 98]))),
                        "cls": cname, "score": float(len(Y))})
    return out


def to_world(boxes, meta):
    th = meta["theta"]; c, s = np.cos(th), np.sin(th); R = np.array([[c, -s], [s, c]]); ctr = np.array(meta["centre"])
    for b in boxes:
        w = R @ np.array([b["cx"], b["cy"]]) + ctr
        b["cx"], b["cy"], b["yaw"] = float(w[0]), float(w[1]), float(b["yaw"] + th)
    return boxes
