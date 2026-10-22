"""Quality check for the UREx LiGrip exports.

For each mission: read every LAZ tile in full (integrity), collect stats, and
render a top-down colour view plus a wall-height slice coloured by capture
time (drift / ghosting check). Usage:
    python qc.py <laz_dir> <traj_list.json> <out_dir> [mission ...]
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAZ_DIR, TRAJ_LIST, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
KEEP = 20          # keep every 20th point for renders
CELL = 0.25        # footprint grid (m)
PX = 0.08          # render pixel size (m)
MAXPX = 4000


def load_traj(path):
    rows = np.loadtxt(path, delimiter=",", comments="#", usecols=(0, 7, 8, 9))
    return rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]


def qc_mission(args):
    mission, traj_path = args
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(LAZ_DIR, f"{mission}_*.laz")))
    n = 0
    zero_rgb = 0
    dark = 0
    gmin, gmax = np.inf, -np.inf
    xs, ys, zs, gs, rgb = [], [], [], [], []
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    bad = []
    for f in files:
        try:
            with laspy.open(f, laz_backend=laspy.LazBackend.LazrsParallel) as r:
                mins = np.minimum(mins, r.header.mins)
                maxs = np.maximum(maxs, r.header.maxs)
                for c in r.chunk_iterator(4_000_000):
                    n += len(c)
                    R, G, B = np.asarray(c.red), np.asarray(c.green), np.asarray(c.blue)
                    s = R.astype(np.uint32) + G + B
                    zero_rgb += int((s == 0).sum())
                    dark += int((s < 3 * 2560).sum())  # < 10 % brightness
                    gt = np.asarray(c.gps_time)
                    gmin, gmax = min(gmin, gt.min()), max(gmax, gt.max())
                    sl = slice(None, None, KEEP)
                    xs.append(np.asarray(c.x)[sl]); ys.append(np.asarray(c.y)[sl])
                    zs.append(np.asarray(c.z)[sl]); gs.append(gt[sl])
                    rgb.append(np.stack([R[sl], G[sl], B[sl]], 1))
        except Exception as e:  # noqa: BLE001
            bad.append(f"{os.path.basename(f)}: {e}")
    x, y, z, g = (np.concatenate(a) for a in (xs, ys, zs, gs))
    col = np.concatenate(rgb).astype(np.float32) / 65535.0

    # footprint and density
    ix = np.floor((x - mins[0]) / CELL).astype(np.int64)
    iy = np.floor((y - mins[1]) / CELL).astype(np.int64)
    _, cc = np.unique(ix * 1_000_000 + iy, return_counts=True)
    good = cc >= 5                      # >= ~1600 pts/m2 at full resolution
    area = int(good.sum()) * CELL * CELL
    density = float(np.median(cc[good])) * KEEP / (CELL * CELL) if good.any() else 0

    # trajectory
    tt, tx, ty, tz = load_traj(traj_path)
    step = np.hypot(np.diff(tx), np.diff(ty))
    traj_len = float(np.sum(np.sqrt(step ** 2 + np.diff(tz) ** 2)))
    # wall slice relative to scanner height at capture time
    sx, sy, sz = (np.interp(g, tt, a) for a in (tx, ty, tz))
    dz = z - sz
    dist = np.hypot(x - sx, y - sy)
    wall = (dz > -0.9) & (dz < -0.1) & (dist < 12)

    # renders: trajectory bbox + 15 m
    lo = np.maximum(mins[:2], [tx.min() - 15, ty.min() - 15])
    hi = np.minimum(maxs[:2], [tx.max() + 15, ty.max() + 15])
    mins = np.array([lo[0], lo[1], mins[2]]); maxs = np.array([hi[0], hi[1], maxs[2]])
    inw = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
    w = min(int((maxs[0] - mins[0]) / PX) + 1, MAXPX)
    h = min(int((maxs[1] - mins[1]) / PX) + 1, MAXPX)
    px = (maxs[0] - mins[0]) / (w - 1 if w > 1 else 1)
    py = (maxs[1] - mins[1]) / (h - 1 if h > 1 else 1)
    pxi = np.clip(((x - mins[0]) / px).astype(np.int64), 0, w - 1)
    pyi = np.clip(((maxs[1] - y) / py).astype(np.int64), 0, h - 1)
    flat = np.where(inw, pyi * w + pxi, w * h)
    cnt = np.bincount(flat, minlength=w * h + 1)[: w * h].astype(np.float32)
    img = np.zeros((w * h, 3), np.float32)
    for k in range(3):
        img[:, k] = np.bincount(flat, weights=col[:, k], minlength=w * h + 1)[: w * h]
    img = img / np.maximum(cnt, 1)[:, None]
    a = np.clip(cnt / 3.0, 0, 1)[:, None]
    img = (np.clip(img * 1.1, 0, 1) * a + 0.08 * (1 - a)).reshape(h, w, 3)
    plt.imsave(os.path.join(OUT, f"{mission}_rgb.png"), img)

    tnorm = (g[wall] - g.min()) / max(np.ptp(g), 1)
    fw = flat[wall]
    tsum = np.bincount(fw, weights=tnorm, minlength=w * h + 1)[: w * h]
    tc = np.bincount(fw, minlength=w * h + 1)[: w * h]
    tmean = np.where(tc > 0, tsum / np.maximum(tc, 1), np.nan).reshape(h, w)
    alpha = np.clip(tc.reshape(h, w) / 2.0, 0, 1)
    rgba = plt.cm.turbo(np.nan_to_num(tmean))
    rgba[..., 3] = alpha
    bg = np.ones((h, w, 4)); bg[..., :3] = 0.08
    comp = bg.copy()
    comp[..., :3] = rgba[..., :3] * alpha[..., None] + bg[..., :3] * (1 - alpha[..., None])
    fig = plt.figure(figsize=(w / 200, h / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(comp[..., :3], interpolation="nearest")
    ax.plot((tx - mins[0]) / px, (maxs[1] - ty) / py, color="white", lw=0.15, alpha=0.35)
    fig.savefig(os.path.join(OUT, f"{mission}_wallslice.png"), dpi=200)
    plt.close(fig)

    res = {
        "mission": mission, "tiles": len(files), "points": n, "bad_tiles": bad,
        "extent_m": [round(float(v), 1) for v in (maxs - mins)],
        "z_range_m": [round(float(mins[2]), 1), round(float(maxs[2]), 1)],
        "footprint_m2": round(area), "density_pts_per_m2": round(density),
        "uncoloured_pct": round(100 * zero_rgb / n, 3), "dark_pct": round(100 * dark / n, 2),
        "gps_span_s": round(float(gmax - gmin), 1), "traj_span_s": round(float(tt[-1] - tt[0]), 1),
        "traj_len_m": round(traj_len, 1), "traj_z_range_m": round(float(tz.max() - tz.min()), 1),
        "wall_slice_pts": int(wall.sum()) * KEEP, "seconds": round(time.time() - t0),
        "render_px_m": round(float(px), 3),
    }
    with open(os.path.join(OUT, f"{mission}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    trajs = json.load(open(TRAJ_LIST))
    todo = sys.argv[4:] or sorted(trajs)
    todo = [m for m in todo if not os.path.exists(os.path.join(OUT, f"{m}.json"))]
    with Pool(3) as pool:
        out = pool.map(qc_mission, [(m, trajs[m]) for m in todo], chunksize=1)
    print("DONE", len(out))
