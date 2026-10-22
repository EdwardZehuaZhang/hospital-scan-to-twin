"""SpatialLM inference without the streaming thread (so errors surface), for one or more PLY files.
python slm_infer.py <model_path> <out_dir> <detect_type: all|object|arch> <seed> <ply> [<ply> ...] [--cat a b c]"""
import os
import sys
import traceback

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, "/opt/work/SpatialLM")
os.chdir("/opt/work/SpatialLM")
from inference import DETECT_TYPE_PROMPT, preprocess_point_cloud  # noqa: E402
from spatiallm import Layout  # noqa: E402
from spatiallm.pcd import cleanup_pcd, get_points_and_colors, load_o3d_pcd  # noqa: E402

args = sys.argv[1:]
cats = []
if "--cat" in args:
    i = args.index("--cat"); cats = args[i + 1:]; args = args[:i]
model_path, out_dir, detect_type, seed, plys = args[0], args[1], args[2], int(args[3]), args[4:]
VOXEL = float(os.environ.get("SLM_VOXEL", "0.05")); MAXPTS = int(os.environ.get("SLM_MAXPTS", "250000"))
os.makedirs(out_dir, exist_ok=True)
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model.to("cuda"); model.set_point_backbone_dtype(torch.float32); model.eval()
num_bins = model.config.point_config["num_bins"]
template = open("code_template.txt").read()
TILE = float(os.environ.get("SLM_TILE", "0"))  # metres; 0 = whole area in one pass


def infer_pc(pc):
    """Run SpatialLM on one Open3D point cloud (frame unchanged); returns Bbox lines in that frame and raw char count."""
    v = VOXEL
    pc0 = pc; pc = pc0.voxel_down_sample(v)
    while len(pc.points) > MAXPTS:
        v *= 1.15; pc = pc0.voxel_down_sample(v)
    grid = Layout.get_grid_size(num_bins)
    pc = cleanup_pcd(pc, voxel_size=grid)
    pts, cols = get_points_and_colors(pc)
    if len(pts) < 2000:
        return [], 0
    mn = np.min(pts, axis=0)
    inp = preprocess_point_cloud(pts, cols, grid, num_bins)
    task = DETECT_TYPE_PROMPT[detect_type]
    if detect_type != "arch" and cats:
        task = task.replace("boxes", ", ".join(cats))
    prompt = f"<|point_start|><|point_pad|><|point_end|>{task} The reference code is as followed: {template}"
    conv = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to("cuda")
    set_seed(seed)
    with torch.no_grad():
        outp = model.generate(input_ids=ids, point_clouds=inp, max_new_tokens=4096, do_sample=True, use_cache=True,
                              temperature=0.6, top_p=0.95, top_k=10, num_beams=1)
    text = tok.decode(outp[0][ids.shape[1]:], skip_special_tokens=True)
    lay = Layout(text); lay.undiscretize_and_unnormalize(num_bins=num_bins); lay.translate(mn)
    lines = [l for l in lay.to_language_string().splitlines() if "=Bbox(" in l]
    return lines, len(text)


for ply in plys:
    try:
        pc = load_o3d_pcd(ply)
        name = os.path.splitext(os.path.basename(ply))[0]
        if TILE <= 0:
            lines, nch = infer_pc(pc); info = f"chars {nch}"
        else:
            P = np.asarray(pc.points); lo, hi = P.min(0), P.max(0); lines = []; info = []
            step = TILE - 1.0
            for x0 in np.arange(lo[0], hi[0], step):
                for y0 in np.arange(lo[1], hi[1], step):
                    m = (P[:, 0] >= x0) & (P[:, 0] < x0 + TILE) & (P[:, 1] >= y0) & (P[:, 1] < y0 + TILE)
                    if m.sum() < 5000:
                        continue
                    sub = pc.select_by_index(np.where(m)[0])
                    tl, nch = infer_pc(sub)
                    # keep boxes centred in this tile's core (0.5 m margin) so seams do not duplicate objects
                    for l in tl:
                        vals = l.split("(", 1)[1].rstrip(")").split(",")
                        cx, cy = float(vals[1]), float(vals[2])
                        if x0 + 0.5 <= cx < x0 + TILE - 0.5 and y0 + 0.5 <= cy < y0 + TILE - 0.5:
                            lines.append(l)
                    info.append(len(tl))
            info = f"tiles {info}"
        lines = [f"bbox_{i}={l.split('=', 1)[1]}" for i, l in enumerate(lines)]
        open(os.path.join(out_dir, f"{name}.txt"), "w").write("\n".join(lines) + "\n")
        print("OK", ply, "boxes", len(lines), info, flush=True)
    except Exception:
        traceback.print_exc()
