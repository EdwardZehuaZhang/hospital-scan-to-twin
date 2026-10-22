# Scanning a hospital that cannot close

Code, annotations and results for a NUS Undergraduate Research Experience (CDE2605R) study of handheld SLAM LiDAR
capture in an operating hospital (National University Hospital, Singapore), with a campus baseline, and of turning
the scan into an editable object-level twin.

The study has four parts, each with code here:

1. **Capture and processing evaluation** of a GreenValley LiGrip O2 handheld scanner: self-consistency, pass-to-pass
   and cross-mission repeatability, processing cost, and a frame-count limit in the vendor colouriser (shown
   causally: a mission that coloured correctly fails when given more panorama frames).
2. **Scan-to-editable-twin pipeline**: open-vocabulary detection in 360-degree panoramas (Grounding DINO, Segment
   Anything), lifting to LiDAR, single-image 3D generation (Hunyuan3D-2 mini), fitting to the LiDAR points, and
   assembly with a Poisson room shell into one glTF scene with a node per object.
3. **Fidelity evaluation of the generated objects** with exact point-to-mesh distances, F-scores, held-out
   (time-split) evaluation, and controls: fitted box, convex hull, uniform-scale mesh, shuffled mesh (another
   object's photo), and a mesh of the scan itself.
4. **Object benchmark**: 86 hand-annotated objects in three areas, on which SpatialLM (zero-shot, three seeds,
   category prompt, ARKitScenes-tuned), Sonata (zero-shot and head fine-tuned, leave one area out), LiDAR
   clustering and the image-based pipeline are compared.

## Main results

| Finding | Number |
|---|---|
| Pass-to-pass agreement, NUH Renal Centre (2.7 min apart) | median 0.8 cm, 95 % within 2 cm |
| Floor height between passes | within 1 cm in 94 to 95 % of 0.5 m cells |
| Colouriser limit | between 2,464 (works) and 3,070 (fails) panorama frames; 4,221 frames fails on a walk that works at 2,464 |
| Generated mesh vs fitted box, held-out scan points (28 hospital objects) | 9.6 cm vs 5.2 cm (paired difference 3.3 cm, 95 % CI 1.3 to 5.1) |
| Shuffled mesh (wrong photo) vs matched mesh | 1.4 cm worse (CI 0.3 to 2.6) |
| Object benchmark F1, BEV IoU 0.25 (86 objects) | SpatialLM 0.25 to 0.52 by seed; LiDAR clustering 0.35; Sonata 0.22; pipeline 0.28 |

## Layout

| Folder | Contents |
|---|---|
| `pipeline/` | Panorama poses (`poses.py`, `twinlib.py`), detection and lifting (`detect_lift.py`), cleaning, view selection, generation and fitting (`gen_place_v2.py`), shell and assembly (`shell.py`), rendering |
| `evaluation/` | Controls and exact metrics (`robust.py`), visibility split (`vis2.py`), time-split held-out test and floor drift (`pass_split.py`), generation seeds (`seeds.py`), cross-mission registration (`register2.py`), statistics (`stats.py`: Wilcoxon, Hodges-Lehmann, bootstrap, Holm) |
| `benchmark/` | Annotation tools (`code/gt_tiles.py`, `code/gt_snap.py`), export (`code/bench_export.py`), SpatialLM tiled inference (`code/slm_infer.py`), Sonata zero-shot and fine-tune (`code/sonata_bench.py`, `code/finetune_head.py`), LiDAR clustering, `evaluate.py`, ground-truth boxes, all predictions and `results.json` |
| `results/` | Per-object metrics for every pipeline version and area, controls, held-out and seed results, repeatability, label audit, logged drops |
| `data/` | Per-mission processing and quality tables and the scripts that produced them |
| `figures/` | Selected result figures |
| `report_tools/` | Figure script and the Markdown-to-Word builder used for the report |

## Running it

The pipeline and evaluation scripts were run on Windows with Python 3.12, PyTorch 2.11 (CUDA 12.8), Transformers
4.51.3, Diffusers 0.33.1, Open3D 0.20, trimesh, laspy and scikit-learn, on an RTX 5060 Ti (16 GB). SpatialLM and
Sonata were run in Ubuntu 24.04 under WSL2 with PyTorch 2.7.1; `benchmark/code/wsl_setup.sh` builds that environment
and `benchmark/code/patch_slm.py` disables flash attention, which does not support this GPU. Paths in the scripts
point at the original workstation and need adjusting. Each script's docstring gives its command line.

`benchmark/evaluate.py` and `evaluation/stats.py` run on the included files alone and reproduce the benchmark and
statistics tables.

## Data availability

The raw scans, panoramas, coloured point clouds and generated scenes are not included. The hospital panoramas show
patients, visitors and staff and are personal data; they are held on the lab workstation. The ground-truth boxes,
predictions and per-object metrics here are coordinates and numbers only.

## Licence

Code: MIT (see `LICENSE`). Third-party models (Grounding DINO, Segment Anything, Hunyuan3D-2, SpatialLM, Sonata)
keep their own licences; SpatialLM 1.1 and Sonata weights are CC-BY-NC-4.0.
