"""Top-down view of furniture height (scanner-relative slice) for one mission, with a metre grid.

python furnview.py <laz_glob> <traj> <out.png> [x0 x1 y0 y1]
"""
import glob
import sys

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

lazg, traj, out = sys.argv[1:4]
win = [float(v) for v in sys.argv[4:8]] if len(sys.argv) >= 8 else None
t = np.loadtxt(traj, delimiter=",", comments="#", usecols=(0, 7, 8, 9))
xs, ys, zs, cs = [], [], [], []
keep = 6 if win else 25
for f in sorted(glob.glob(lazg)):
    with laspy.open(f, laz_backend=laspy.LazBackend.LazrsParallel) as r:
        for c in r.chunk_iterator(5_000_000):
            x, y, z = np.asarray(c.x), np.asarray(c.y), np.asarray(c.z)
            m = np.ones(len(x), bool)
            if win:
                m = (x > win[0]) & (x < win[1]) & (y > win[2]) & (y < win[3])
            m[np.arange(len(x)) % keep != 0] = False
            g = np.asarray(c.gps_time)[m]
            sz = np.interp(g, t[:, 0], t[:, 3])
            dz = z[m] - sz
            sel = (dz > -1.45) & (dz < 0.1)  # handheld ~1.4-1.6 m above floor: floor+0.1 .. ~1.6 m
            xs.append(x[m][sel]); ys.append(y[m][sel]); zs.append(dz[sel])
            cs.append(np.stack([np.asarray(c.red)[m][sel], np.asarray(c.green)[m][sel], np.asarray(c.blue)[m][sel]], 1))
x, y, dz = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
col = np.clip(np.concatenate(cs) / 65535.0 * 1.1, 0, 1)
o = np.argsort(dz)
fig, ax = plt.subplots(figsize=(16, 16), dpi=110)
ax.set_facecolor("#111")
ax.scatter(x[o], y[o], c=col[o], s=0.3 if win else 0.05, linewidths=0)
if not win:
    ax.plot(t[:, 1], t[:, 2], color="cyan", lw=0.3, alpha=0.5)
else:
    mt = (t[:, 1] > win[0]) & (t[:, 1] < win[1]) & (t[:, 2] > win[2]) & (t[:, 2] < win[3])
    ax.plot(t[mt, 1], t[mt, 2], ".", color="cyan", ms=0.5, alpha=0.5)
    ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
ax.set_aspect("equal")
step = 2 if win else 10
ax.set_xticks(np.arange(np.floor(ax.get_xlim()[0] / step) * step, ax.get_xlim()[1], step))
ax.set_yticks(np.arange(np.floor(ax.get_ylim()[0] / step) * step, ax.get_ylim()[1], step))
ax.grid(color="#444", lw=0.4)
ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(out)
print("points", len(x))
