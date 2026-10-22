"""Object benchmark evaluation: predicted oriented boxes against hand-annotated LiDAR boxes, in bird's-eye view.

Matching is greedy by BEV IoU (oriented rectangles); a prediction counts as a true positive at IoU >= 0.25. Reported
per method: precision, recall, F1 (class-agnostic), class-aware recall, and for matched pairs the median centre error
and median footprint size error. Ground truth and predictions are restricted to the region each method could see
(the area window intersected with the 25 m export crop).

python evaluate.py     (reads benchmark/gt, benchmark/meta, benchmark/pred/<method>/<area>.{json,txt}; writes results.json)
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AREAS = {"lobby": (22, 40, -68, -50), "corridor": (76, 102, -137, -114), "campus": (66, 86, -62, -36)}
SEATING = {"chair", "sofa", "stool", "bench", "armchair", "couch", "seat", "ottoman", "chairs"}
TABLE = {"table", "desk", "coffee_table", "dining_table", "side_table", "end_table", "counter"}
PLANT = {"plant", "potted_plant", "flower", "tree"}
NON_FURNITURE = {"curtain", "painting", "picture", "window", "door", "mirror", "light", "lamp", "chandelier", "tv",
                 "television", "clock", "air_conditioner", "fan", "blinds", "frame", "poster", "whiteboard", "board"}


def cls_map(c):
    c = c.lower().strip().replace(" ", "_")
    if c in SEATING or "chair" in c or "sofa" in c or "bench" in c:
        return "seating"
    if c in TABLE or "table" in c:
        return "table"
    if c in PLANT or "plant" in c:
        return "plant"
    if c in ("kiosk", "sign"):
        return c
    return "other"


def rect(b):
    R = np.array([[np.cos(b["yaw"]), -np.sin(b["yaw"])], [np.sin(b["yaw"]), np.cos(b["yaw"])]])
    return (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [b["sx"] / 2, b["sy"] / 2]) @ R.T + [b["cx"], b["cy"]]


def clip(subject, clipper):
    """Sutherland-Hodgman polygon clipping (both convex, counter-clockwise)."""
    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            if inside(q):
                if not inside(p):
                    out.append(inter(p, q, a, b))
                out.append(q)
            elif inside(p):
                out.append(inter(p, q, a, b))
    return out


def inter(p, q, a, b):
    x1, y1, x2, y2, x3, y3, x4, y4 = *p, *q, *a, *b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return q
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = np.array(poly).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def iou(a, b):
    ra, rb = rect(a), rect(b)
    ia = area(clip([tuple(p) for p in ra], [tuple(p) for p in rb]))
    u = a["sx"] * a["sy"] + b["sx"] * b["sy"] - ia
    return ia / u if u > 0 else 0.0


def load_pred(path, meta):
    if path.endswith(".json"):
        return json.load(open(path))
    th = meta["theta"]; c, s = np.cos(th), np.sin(th); R = np.array([[c, -s], [s, c]]); ctr = np.array(meta["centre"])
    out = []
    for line in open(path):
        m = re.match(r"\s*bbox_\d+=Bbox\(([^,]+),(.*)\)", line)
        if not m:
            continue
        x, y, z, ang, sx, sy, sz = map(float, m.group(2).split(","))
        if m.group(1).strip().lower() in NON_FURNITURE:
            continue
        w = R @ np.array([x, y]) + ctr
        out.append({"cx": float(w[0]), "cy": float(w[1]), "sx": sx, "sy": sy, "yaw": ang + th, "z0": z - sz / 2, "sz": sz,
                    "cls": m.group(1).strip(), "score": 1.0})
    return out


def in_region(b, a, meta):
    x0, x1, y0, y1 = AREAS[a]
    if not (x0 <= b["cx"] <= x1 and y0 <= b["cy"] <= y1):
        return False
    th = meta["theta"]; c, s = np.cos(th), np.sin(th); R = np.array([[c, -s], [s, c]])
    loc = (np.array([b["cx"], b["cy"]]) - np.array(meta["centre"])) @ R
    return bool(np.all(np.abs(loc) < 12.5))


def evaluate(gt, pred, thr=0.25):
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)), reverse=True)
    ug, up, matches = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in ug or j in up:
            continue
        ug.add(i); up.add(j); matches.append((i, j, v))
    tp = len(matches)
    ce = [np.hypot(gt[i]["cx"] - pred[j]["cx"], gt[i]["cy"] - pred[j]["cy"]) for i, j, _ in matches]
    se = [np.mean(np.abs(np.sort([gt[i]["sx"], gt[i]["sy"]]) - np.sort([pred[j]["sx"], pred[j]["sy"]]))) for i, j, _ in matches]
    same = sum(gt[i]["cls"] == cls_map(pred[j]["cls"]) for i, j, _ in matches)
    per = {}
    for c in ("seating", "table", "kiosk", "sign", "plant", "other"):
        n = sum(g["cls"] == c for g in gt)
        if n:
            per[c] = (sum(gt[i]["cls"] == c for i, j, _ in matches), n)
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp, "same_class": same, "centre_err": ce, "size_err": se,
            "iou": [v for _, _, v in matches], "per_class": per}


def summarise(rs):
    tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    ce = [x for r in rs for x in r["centre_err"]]; se = [x for r in rs for x in r["size_err"]]
    per = {}
    for r in rs:
        for c, (m, n) in r["per_class"].items():
            a, b = per.get(c, (0, 0)); per[c] = (a + m, b + n)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0.0,
            "class_aware_recall": sum(r["same_class"] for r in rs) / (tp + fn) if tp + fn else 0.0,
            "median_centre_err_m": float(np.median(ce)) if ce else None, "median_size_err_m": float(np.median(se)) if se else None,
            "median_iou": float(np.median([x for r in rs for x in r["iou"]])) if tp else None,
            "per_class_recall": {c: (m / n if n else None, n) for c, (m, n) in per.items()}}


if __name__ == "__main__":
    meta = {a: json.load(open(os.path.join(HERE, "meta", f"{a}_meta.json"))) for a in AREAS}
    gt = {a: [b for b in json.load(open(os.path.join(HERE, "gt", a, "gt_boxes.json"))) if b["n_pts"] >= 80 and in_region(b, a, meta[a])]
          for a in AREAS}
    print("ground truth", {a: len(v) for a, v in gt.items()})
    res = {"n_gt": {a: len(v) for a, v in gt.items()}, "methods": {}}
    for mdir in sorted(glob.glob(os.path.join(HERE, "pred", "*"))):
        m = os.path.basename(mdir); per_area = {}
        for a in AREAS:
            f = [p for p in glob.glob(os.path.join(mdir, a + ".*"))]
            if not f:
                continue
            pred = [p for p in load_pred(f[0], meta[a]) if in_region(p, a, meta[a])]
            per_area[a] = {thr: evaluate(gt[a], pred, thr) for thr in (0.25, 0.1)}
        if not per_area:
            continue
        res["methods"][m] = {"areas": {a: {"0.25": summarise([v[0.25]]), "n_pred": v[0.25]["tp"] + v[0.25]["fp"]} for a, v in per_area.items()},
                             "all@0.25": summarise([v[0.25] for v in per_area.values()]),
                             "all@0.10": summarise([v[0.1] for v in per_area.values()])}
        s = res["methods"][m]["all@0.25"]
        print(f"{m:22s} P {s['precision']:.2f} R {s['recall']:.2f} F1 {s['f1']:.2f} cls-R {s['class_aware_recall']:.2f} "
              f"centre {s['median_centre_err_m']} size {s['median_size_err_m']}  |  " +
              "  ".join(f"{a}: F1 {v['0.25']['f1']:.2f} ({v['n_pred']} pred)" for a, v in res["methods"][m]["areas"].items()))
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
