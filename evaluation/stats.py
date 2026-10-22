"""Statistics for the twin results: medians, paired Wilcoxon tests, Hodges-Lehmann differences with bootstrap
intervals and Holm correction, for the control comparison (robust.csv), the held-out test (heldout.csv) and the
seed variance (seeds.csv). Writes stats.json and prints Markdown tables.

python3 stats.py
"""
import csv
import json
import os

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "twin-pilot", "results")
AREAS = [("Lobby", "v3-lobby"), ("Renal Centre", "v3-corridor"), ("Campus", "v3-campus")]
rng = np.random.default_rng(0)


def load(name):
    out = {}
    for area, d in AREAS:
        p = os.path.join(RES, d, name)
        if os.path.exists(p):
            out[area] = list(csv.DictReader(open(p)))
    return out


def hl(d):
    """Hodges-Lehmann estimate of the median paired difference (median of Walsh averages)."""
    d = np.asarray(d); w = (d[:, None] + d[None, :]) / 2
    return float(np.median(w[np.triu_indices(len(d))]))


def boot(d, n=4000):
    d = np.asarray(d)
    b = [hl(d[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k]); adj[k] = min(1.0, run)
    return adj.tolist()


def compare(rows, a, b, metric):
    x = np.array([float(r[f"{a}_{metric}"]) for r in rows]); y = np.array([float(r[f"{b}_{metric}"]) for r in rows])
    d = x - y
    p = float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0
    lo, hi = boot(d)
    return {"n": len(d), "median_a": float(np.median(x)), "median_b": float(np.median(y)), "hl": hl(d),
            "ci": [lo, hi], "p": p}


S = {}
R = load("robust.csv")
metrics = [("mean_cm", "Scan to model (cm)", -1), ("within5_pct", "Within 5 cm (%)", 1), ("F_5cm", "F-score at 5 cm", 1),
           ("F_5pctdiag", "F-score at 5 % of diagonal", 1), ("invented_pct", "Unsupported surface (%)", -1),
           ("invented_visible_pct", "Unsupported, in line of sight (%)", -1),
           ("invented_unseen_pct", "Unsupported, never in line of sight (%)", -1)]
models = ["gen", "gen_unif", "shuffled", "box", "hull", "scanmesh"]
if R:
    hosp = [r for a in ("Lobby", "Renal Centre") for r in R.get(a, [])]
    groups = {**R, "Hospital (both areas)": hosp}
    S["medians"] = {g: {m: {k: float(np.median([float(r[f"{m}_{k}"]) for r in rows])) for k, _, _ in metrics}
                        for m in models} for g, rows in groups.items() if rows}
    S["n"] = {g: len(rows) for g, rows in groups.items()}
    comps = {}
    for g, rows in groups.items():
        if len(rows) < 6:
            continue
        cc = {}
        for other in ("box", "hull", "shuffled", "gen_unif"):
            for k, _, _ in metrics[:5]:
                cc[f"gen_vs_{other}_{k}"] = compare(rows, "gen", other, k)
        adj = holm([v["p"] for v in cc.values()])
        for (kk, v), a in zip(cc.items(), adj):
            v["p_holm"] = a
        comps[g] = cc
    S["comparisons"] = comps

H = load("heldout.csv")
if H:
    rows = [r for v in H.values() for r in v]
    S["heldout"] = {"n": len(rows)}
    for m in ("gen", "box"):
        S["heldout"][m] = {k: float(np.median([float(r[f"{m}_{k}"]) for r in rows])) for k in
                           ("insample_mean_cm", "heldout_mean_cm", "heldout_within5_pct", "unsupported_by_A_pct",
                            "unsupported_seen_by_B_pct")}
    if len(rows) >= 6:
        S["heldout"]["gen_vs_box_heldout_mean"] = compare(rows, "gen", "box", "heldout_mean_cm")
        S["heldout"]["gen_vs_box_heldout_within5"] = compare(rows, "gen", "box", "heldout_within5_pct")

SD = load("seeds.csv")
if SD:
    rows = [r for v in SD.values() for r in v]
    m = np.array([[float(r[f"s{s}_mean_cm"]) for s in (0, 1, 2)] for r in rows])
    u = np.array([[float(r[f"s{s}_invented_pct"]) for s in (0, 1, 2)] for r in rows])
    S["seeds"] = {"n": len(rows), "median_range_mean_cm": float(np.median(m.max(1) - m.min(1))),
                  "p90_range_mean_cm": float(np.percentile(m.max(1) - m.min(1), 90)),
                  "median_sd_mean_cm": float(np.median(m.std(1, ddof=1))),
                  "median_range_invented_pct": float(np.median(u.max(1) - u.min(1))),
                  "median_of_seed_medians_cm": [float(np.median(m[:, s])) for s in range(3)]}

json.dump(S, open(os.path.join(HERE, "stats.json"), "w"), indent=1)

# ---- print tables ----
if "medians" in S:
    names = {"gen": "Generated", "gen_unif": "Generated, uniform scale", "shuffled": "Shuffled mesh (wrong photo)",
             "box": "Oriented box (fitted)", "hull": "Convex hull", "scanmesh": "Mesh of the scan (upper bound)"}
    for g, med in S["medians"].items():
        print(f"\n{g} (n = {S['n'][g]})\n| Model | " + " | ".join(l for _, l, _ in metrics) + " |")
        print("|---" * (len(metrics) + 1) + "|")
        for m in models:
            print(f"| {names[m]} | " + " | ".join(f"{med[m][k]:.1f}" for k, _, _ in metrics) + " |")
for g, cc in S.get("comparisons", {}).items():
    print(f"\n{g}: generated minus other (HL [95% CI], p, p_holm)")
    for k, v in cc.items():
        print(f"  {k}: {v['hl']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] p={v['p']:.4f} holm={v['p_holm']:.4f}")
print("\nheldout", json.dumps(S.get("heldout"), indent=1))
print("\nseeds", json.dumps(S.get("seeds"), indent=1))
