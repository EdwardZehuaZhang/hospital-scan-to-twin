"""Figures for the UREx report. Run: python3 make_figs.py  (writes fig*.png next to this file)."""
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # nus-urex-paper
DATA = os.path.join(ROOT, "data")
TWIN = os.path.join(ROOT, "twin-pilot", "results")
SRC = os.path.join(HERE, "src")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight"})
C_HOSP, C_CAMP, C_GREY, C_ACC = "#c0392b", "#2471a3", "#7f8c8d", "#e67e22"


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
    stages = [
        ("Capture", "LiGrip O2 handheld\nLiDAR + 360 camera\n14 walks, 5.7 h", 0.1),
        ("Process", "Offline SLAM,\nfilter, colourise\n(2 m frame spacing)", 2.05),
        ("Check", "Integrity, drift,\ncolour, repeatability\nbetween walks", 4.0),
        ("Perceive", "Open-vocabulary\ndetection + SAM in\npanoramas, lift to LiDAR", 5.95),
        ("Generate", "Image-to-3D per object,\nfit to LiDAR, colour\nfrom scan; room shell", 7.9),
    ]
    for i, (t, d, x) in enumerate(stages):
        col = C_CAMP if i < 3 else C_ACC
        ax.add_patch(FancyBboxPatch((x, 0.9), 1.75, 1.9, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=col, ec="none", alpha=0.12))
        ax.text(x + 0.875, 2.5, t, ha="center", va="center", fontsize=11, weight="bold", color=col)
        ax.text(x + 0.875, 1.6, d, ha="center", va="center", fontsize=8.2)
        if i < 4:
            ax.annotate("", xy=(x + 1.95, 1.85), xytext=(x + 1.78, 1.85),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2))
    ax.text(2.95, 0.45, "Reality capture (geometry is measured)", ha="center", color=C_CAMP, fontsize=9)
    ax.text(7.85, 0.45, "Generation anchored to the scan (geometry is generated, then checked)", ha="center",
            color=C_ACC, fontsize=9)
    ax.text(9.65, 0.1, "Output: editable twin (GLB),\none node per object", ha="right", fontsize=8, color="#333")
    fig.savefig(os.path.join(HERE, "fig1_pipeline.png")); plt.close(fig)


def fig_sites():
    ims = [("20260719103709_rgb.png", "(a) NUH Main Building interior\n41 min, 2.48 km"),
           ("20260719112952_rgb.png", "(b) NUH exterior loop\n21 min, 1.62 km"),
           ("20260717074326_rgb.png", "(c) Campus: SDE4 to SDE2\n32 min, 1.73 km")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, (f, t) in zip(axs, ims):
        im = Image.open(os.path.join(SRC, f)).convert("RGB"); im.thumbnail((1400, 1400))
        ax.imshow(im); ax.set_title(t, fontsize=9); ax.axis("off")
    fig.savefig(os.path.join(HERE, "fig2_sites.png"), facecolor="white"); plt.close(fig)


def fig_colour():
    q = list(csv.DictReader(open(os.path.join(DATA, "quality-results.csv"))))
    fixed = {"20260715143411": 825, "20260716114515": 767, "20260717085402": 970, "20260718103948": 965,
             "20260719103709": 1008}
    fig = plt.figure(figsize=(11, 4.0))
    ax = fig.add_axes([0.06, 0.15, 0.36, 0.75])
    for r in q:
        f, u = int(r["camera_frames"]), float(r["uncoloured_pct"])
        hosp = r["site"] == "NUH"
        ax.scatter(f, u, s=36, color=C_HOSP if hosp else C_CAMP, zorder=3, edgecolor="white", lw=0.6)
        if r["mission"] in fixed:
            ax.annotate("", xy=(fixed[r["mission"]], 0), xytext=(f, u),
                        arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=0.8, alpha=0.7))
            ax.scatter(fixed[r["mission"]], 0, s=36, facecolor="none", edgecolor=C_HOSP if hosp else C_CAMP, zorder=3)
    ax.axvspan(2464, 3070, color="#999", alpha=0.18, lw=0)
    ax.text(2767, 6.5, "limit lies\nin this band\n(2,464 to\n3,070)", fontsize=7.5, color="#555", ha="center")
    ax.set_xlabel("Panorama frames used for colourisation"); ax.set_ylabel("Points with no colour (%)")
    ax.set_title("(a) Vendor colouriser fails on long walks;\nre-run with 2 m spacing (arrows) fixes all five", fontsize=9)
    ax.scatter([], [], color=C_HOSP, label="NUH"); ax.scatter([], [], color=C_CAMP, label="Campus")
    ax.scatter([], [], facecolor="none", edgecolor="#333", label="after fix"); ax.legend(frameon=False, fontsize=8)
    before = Image.open(os.path.join(DATA, "qc", "colour-views.png")).convert("RGB")
    after = Image.open(os.path.join(DATA, "qc", "colour-views-fixed.png")).convert("RGB")
    # tile 10 (UTown part 1) in a 4-column sheet of 460 x 485 tiles
    box = (2 * 460, 2 * 485 + 25, 3 * 460, 3 * 485)
    for i, (im, t) in enumerate([(before, "(b) UTown part 1, vendor default\n3,070 frames: 16.8 % uncoloured"),
                                 (after, "(c) Same walk, 2 m frame spacing\n970 frames: 0 % uncoloured")]):
        a = fig.add_axes([0.46 + i * 0.27, 0.08, 0.26, 0.82]); a.imshow(im.crop(box)); a.axis("off"); a.set_title(t, fontsize=9)
    fig.savefig(os.path.join(HERE, "fig4_colour.png"), facecolor="white"); plt.close(fig)


def fig_repeat():
    d = os.path.join(TWIN, "repeat")
    if not os.path.exists(d):
        return
    sets = [("NUH Renal Centre,\npass 1 vs 2 (2.7 min)", "pass_corridor_dist.npy", C_HOSP),
            ("NUH lobby,\npass 1 vs later (to 24 min)", "pass_lobby_dist.npy", "#e74c3c"),
            ("UTown entrance,\nday 1 vs day 2", "reg2_20260717085402_20260718103948_dist.npy", C_CAMP)]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    meds, p90s, labs, cols = [], [], [], []
    for lab, f, c in sets:
        x = 100 * np.load(os.path.join(d, f)).astype(float)
        axs[0].hist(np.clip(x, 0, 30), bins=np.linspace(0, 30, 61), density=True, histtype="step", lw=1.6, color=c,
                    label=lab.replace("\n", " "))
        meds.append(np.median(x)); p90s.append(np.percentile(x, 90)); labs.append(lab); cols.append(c)
    axs[0].set_yscale("log"); axs[0].set_xlabel("Distance to the earlier scan (cm), points within 30 cm")
    axs[0].set_ylabel("Density (log)"); axs[0].legend(frameon=False, fontsize=7)
    axs[0].set_title("(a) Distribution of distances", fontsize=9)
    x = np.arange(len(labs))
    axs[1].bar(x, p90s, color=cols, alpha=0.3, label="90th percentile")
    axs[1].bar(x, meds, color=cols, alpha=0.9, label="median")
    for i, (m, q) in enumerate(zip(meds, p90s)):
        axs[1].text(i, q + 0.3, f"{q:.1f}", ha="center", fontsize=7); axs[1].text(i, m + 0.3, f"{m:.1f}", ha="center", fontsize=7, color="white" if m > 2 else "#222")
    axs[1].axhline(3, color="#999", ls=":", lw=0.8); axs[1].text(2.45, 3.2, "median limit set in advance", fontsize=6.5, ha="right", color="#777")
    axs[1].axhline(10, color="#999", ls="--", lw=0.8); axs[1].text(2.45, 10.2, "90th percentile limit", fontsize=6.5, ha="right", color="#777")
    axs[1].set_xticks(x); axs[1].set_xticklabels(labs, fontsize=6.5); axs[1].set_ylabel("cm")
    axs[1].legend(frameon=False, fontsize=7, loc="upper left"); axs[1].set_title("(b) Median and 90th percentile", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig3_repeatability.png"), facecolor="white"); plt.close(fig)

def load_metrics():
    out = {}
    for area, d in [("Lobby (NUH)", "v3-lobby"), ("Renal Centre (NUH)", "v3-corridor"), ("Campus (SDE)", "v3-campus")]:
        p = os.path.join(TWIN, d, "metrics.csv")
        if os.path.exists(p):
            out[area] = list(csv.DictReader(open(p)))
    return out


def fig_object_metrics():
    M = load_metrics()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    cols = {"Lobby (NUH)": C_HOSP, "Renal Centre (NUH)": "#e74c3c", "Campus (SDE)": C_CAMP}
    for j, (k, lab) in enumerate([("scan_to_mesh_mean_cm", "Scan to generated mesh, mean (cm)"),
                                  ("scan_within_5cm_pct", "Scan points within 5 cm of mesh (%)"),
                                  ("mesh_unsupported_pct", "Invented surface (%)")]):
        ax = axs[j]
        for i, (area, rows) in enumerate(M.items()):
            v = np.array([float(r[k]) for r in rows])
            x = i + np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v, s=18, color=cols[area], alpha=0.75, edgecolor="white", lw=0.4)
            ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="#222", lw=1.6)
        ax.set_xticks(range(len(M))); ax.set_xticklabels([a.replace(" (", "\n(") for a in M], fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
    fig.suptitle("Per-object fidelity of generated, LiDAR-anchored models (bar = median)", fontsize=10, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig6_object_metrics.png"), facecolor="white"); plt.close(fig)


def fig_ablation():
    vals = {}
    for v, d in [("v1", "v1-lobby"), ("v2", "v2-lobby"), ("v3", "v3-lobby")]:
        rows = list(csv.DictReader(open(os.path.join(TWIN, d, "metrics.csv"))))
        vals[v] = [np.median([float(r[k]) for r in rows]) for k in
                   ("scan_to_mesh_mean_cm", "scan_within_5cm_pct", "mesh_unsupported_pct")] + [len(rows)]
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
    labels = ["v1\nnearest view,\nuniform scale", "v2\nclean view, SAM,\nsplit, per-axis", "v3\n+ wall filter,\narea clip"]
    for j, (t, c) in enumerate([("Median scan to mesh (cm)", C_ACC), ("Median scan within 5 cm (%)", C_CAMP),
                                ("Median invented surface (%)", C_HOSP)]):
        y = [vals[v][j] for v in ("v1", "v2", "v3")]
        axs[j].bar(range(3), y, color=c, alpha=0.8); axs[j].set_xticks(range(3)); axs[j].set_xticklabels(labels, fontsize=7)
        axs[j].set_title(t, fontsize=9)
        for i, yy in enumerate(y):
            axs[j].text(i, yy, f"{yy:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig5_ablation.png"), facecolor="white"); plt.close(fig)
    return vals


def fig_examples():
    """Rows from the v3 comparison sheets: crop | scan | generated mesh."""
    rows = []
    for d, ids in [("v3-lobby", None), ("v3-corridor", None), ("v3-campus", None)]:
        p = os.path.join(TWIN, d, "compare.jpg")
        if os.path.exists(p):
            im = Image.open(p).convert("RGB"); n = im.height // 280
            rows.append((d, im, n))
    picks = []
    for d, im, n in rows:
        met = list(csv.DictReader(open(os.path.join(TWIN, d, "metrics.csv"))))
        order = np.argsort([float(r["scan_to_mesh_mean_cm"]) for r in met])
        for i in list(order[:2]) + [order[-1]]:  # two best, one worst per area
            picks.append(im.crop((0, i * 280, 768, i * 280 + 280)))
    if not picks:
        return
    sheet = Image.new("RGB", (768 * 3, 280 * ((len(picks) + 2) // 3)), "white")
    for k, p in enumerate(picks):
        sheet.paste(p, ((k % 3) * 768, (k // 3) * 280))
    sheet.save(os.path.join(HERE, "fig7_examples.png"))


def fig_scenes():
    ims = [(os.path.join(TWIN, d, "scene_preview.png"), t) for d, t in
           [("v3-lobby", "(a) NUH lobby"), ("v3-corridor", "(b) NUH Renal Centre corridor"), ("v3-campus", "(c) Campus (SDE)")]
           if os.path.exists(os.path.join(TWIN, d, "scene_preview.png"))]
    fig, axs = plt.subplots(len(ims), 1, figsize=(10, 3.6 * len(ims)))
    axs = np.atleast_1d(axs)
    for ax, (f, t) in zip(axs, ims):
        im = Image.open(f).convert("RGB")
        bb = Image.eval(im.convert("L"), lambda v: 255 if v > 40 else 0).getbbox()  # trim the dark margin
        if bb:
            im = im.crop((max(0, bb[0] - 20), max(0, bb[1] - 20), min(im.width, bb[2] + 20), min(im.height, bb[3] + 20)))
        ax.imshow(im); ax.axis("off"); ax.set_title(t + ": shell in scan colour, generated objects in orange", fontsize=9)
    fig.patch.set_facecolor("white"); fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig9_scenes.png")); plt.close(fig)


def fig_baselines():
    """Generated model against an oriented box and the convex hull fitted to the same points, per object."""
    rows = []
    for area in ("lobby", "corridor"):
        f = os.path.join(TWIN, "baselines", area + ".csv")
        if os.path.exists(f):
            rows += [r for r in csv.DictReader(open(f)) if not (area == "lobby" and r["id"] == "0")]
    if not rows:
        return
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    models = [("box", "Oriented box", C_GREY), ("gen", "Generated", C_ACC), ("hull", "Convex hull", C_CAMP)]
    for j, (m, t) in enumerate([("mean_cm", "Scan to model, mean (cm)"), ("within5_pct", "Scan points within 5 cm (%)"),
                                ("unsupported_pct", "Invented surface (%)")]):
        ax = axs[j]
        V = {k: np.array([float(r[f"{k}_{m}"]) for r in rows]) for k, _, _ in models}
        for a, b in zip(V["box"], V["gen"]):
            ax.plot([0, 1], [a, b], color="#bbb", lw=0.6, zorder=1)
        for a, b in zip(V["gen"], V["hull"]):
            ax.plot([1, 2], [a, b], color="#ddd", lw=0.6, zorder=1)
        for i, (k, lab, c) in enumerate(models):
            ax.scatter(np.full(len(V[k]), i), V[k], s=16, color=c, zorder=2, edgecolor="white", lw=0.4)
            ax.plot([i - 0.25, i + 0.25], [np.median(V[k])] * 2, color="#222", lw=1.8, zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l, _ in models], fontsize=8); ax.set_title(t, fontsize=9)
    fig.suptitle(f"Generated models against geometric baselines on the same LiDAR points ({len(rows)} hospital objects; bar = median)",
                 fontsize=9.5, y=1.03)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig8_baselines.png"), facecolor="white"); plt.close(fig)


def fig_controls():
    """(a) generated mesh against the controls on the hospital objects; (b) held-out generated vs box."""
    rows, held = [], []
    for d in ("v3-lobby", "v3-corridor"):
        f = os.path.join(TWIN, d, "robust.csv")
        if os.path.exists(f):
            rows += list(csv.DictReader(open(f)))
        h = os.path.join(TWIN, d, "heldout.csv")
        if os.path.exists(h):
            held += list(csv.DictReader(open(h)))
    if not rows:
        return
    models = [("box", "Box", C_GREY), ("hull", "Hull", "#95a5a6"), ("gen_unif", "Uniform", "#f5b041"),
              ("shuffled", "Shuffled", "#a569bd"), ("gen", "Generated", C_ACC)]
    fig = plt.figure(figsize=(13, 3.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.8], wspace=0.35)
    for j, (k, t) in enumerate([("mean_cm", "Scan to model (cm)"), ("F_5pctdiag", "F-score at 5 % of diagonal"),
                                ("invented_pct", "Unsupported surface (%)")]):
        ax = fig.add_subplot(gs[0, j])
        for i, (m, lab, c) in enumerate(models):
            v = np.array([float(r[f"{m}_{k}"]) for r in rows])
            ax.scatter(i + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=12, color=c, alpha=0.8, edgecolor="white", lw=0.3)
            ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="#222", lw=1.6)
        ax.set_xticks(range(len(models))); ax.set_xticklabels([l for _, l, _ in models], fontsize=7.5, rotation=35, ha="right"); ax.set_title(t, fontsize=9)
    fig.text(0.05, 0.99, f"(a) Controls, all points ({len(rows)} hospital objects; bar = median)", fontsize=9, va="top")
    if held:
        ax = fig.add_subplot(gs[0, 3])
        g = np.array([float(r["gen_heldout_mean_cm"]) for r in held]); b = np.array([float(r["box_heldout_mean_cm"]) for r in held])
        for a_, b_ in zip(b, g):
            ax.plot([0, 1], [a_, b_], color="#ccc", lw=0.7, zorder=1)
        ax.scatter(np.zeros(len(b)), b, s=12, color=C_GREY, zorder=2); ax.scatter(np.ones(len(g)), g, s=12, color=C_ACC, zorder=2)
        for i, v in enumerate((b, g)):
            ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="#222", lw=1.6, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Box", "Generated"], fontsize=7.5, rotation=35, ha="right")
        ax.set_title("(b) Held out: fit on first pass,\nscore on later passes (cm)", fontsize=9)
    fig.savefig(os.path.join(HERE, "fig8_controls.png"), facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); fig_sites(); fig_colour(); fig_repeat(); fig_object_metrics()
    fig_examples(); fig_scenes(); fig_controls(); print(sorted(glob.glob(os.path.join(HERE, "fig*.png"))))
