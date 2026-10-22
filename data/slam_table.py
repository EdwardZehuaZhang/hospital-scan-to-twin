"""Build a results table from LiDAR360MLS SLAM report PDFs (pdftotext -raw)."""
import csv
import re
import subprocess
import sys
from pathlib import Path

D = Path(sys.argv[1])
SITE = {
    "20260719103709": ("NUH", "NUH interior"),
    "20260719112952": ("NUH", "NUH exterior"),
    "20260715115721": ("SDE", "SDE4 onboarding test 1 (processed 9 Oct on Michael)"),
    "20260715120305": ("SDE", "SDE4 onboarding test 2 (processed 9 Oct on Michael)"),
    "20260715111434": ("NUS", "Indoor test (15 Jul)"),
    "20260717074326": ("SDE", "SDE4 to SDE2"),
    "20260715143411": ("SDE", "Bus stop to SDE3, part 1"),
    "20260715152833": ("SDE", "Bus stop to SDE3, part 2"),
    "20260715163849": ("SDE", "Bus stop to SDE3, part 3"),
    "20260716105000": ("SDE", "SDE1 to SDE4, part 1"),
    "20260716114515": ("SDE", "SDE1 to SDE4, part 2"),
    "20260716125636": ("SDE", "SDE1 to SDE4, part 3"),
    "20260717085402": ("UTown", "UTown entrance, part 1"),
    "20260718103948": ("UTown", "UTown entrance, part 2"),
}


def grab(pat, txt, cast=str):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else ""


rows = []
for pdf in sorted(D.glob("*-SLAM-Report.pdf")):
    mid = pdf.name.split("-")[0]
    txt = subprocess.run(["pdftotext", "-raw", str(pdf), "-"], capture_output=True, text=True).stdout
    acc = re.search(r"Percentage\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", txt)
    site, label = SITE.get(mid, ("?", mid))
    rows.append({
        "site": site, "mission": mid, "label": label,
        "duration_min": grab(r"Collection duration\s+([\d.]+)min", txt),
        "trajectory_m": grab(r"Trajectory length\s+([\d.]+)m", txt),
        "lidar_GB": grab(r"LiDAR file size\s+([\d.]+)GB", txt),
        "camera_GB": grab(r"Camera file size\s+([\d.]+)GB", txt),
        "slam_min": grab(r"SLAM runtime\s+([\d.]+)min", txt),
        "filter_min": grab(r"Filtering runtime\s+([\d.]+)min", txt),
        "colour_min": grab(r"Colorization runtime\s+([\d.]+)min", txt),
        "total_min": grab(r"Total runtime\s+([\d.]+)min", txt),
        "ratio": grab(r"Total runtime\s+[\d.]+min\s*\((1:[\d.]+)\)", txt),
        "acc_0_1cm": acc.group(1) if acc else "", "acc_1_3cm": acc.group(2) if acc else "",
        "acc_3_5cm": acc.group(3) if acc else "", "acc_5_10cm": acc.group(4) if acc else "",
        "acc_10_20cm": acc.group(5) if acc else "", "acc_20cm_plus": acc.group(6) if acc else "",
        "colour_errors": len(re.findall(r"LaunchWeightedColorKernel failed", txt)),
        "images_unusable": len(re.findall(r"Feature extraction\s+result is insufficient", txt)),
        "gpu_cpu_fallbacks": len(re.findall(r"exceeds GPU max_points", txt)),
    })
order = list(SITE)
rows.sort(key=lambda r: order.index(r["mission"]) if r["mission"] in order else 99)
with open(D.parent / "slam-results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r["mission"], r["label"], r["trajectory_m"], r["acc_0_1cm"], r["colour_errors"], r["images_unusable"], r["gpu_cpu_fallbacks"])
