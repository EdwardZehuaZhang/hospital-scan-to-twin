source /opt/conda/etc/profile.d/conda.sh; conda activate slm
B=/mnt/e/WSL/bench
cd /opt/work/SpatialLM
for seed in 0 1 2; do
  SLM_TILE=10 python /mnt/e/WSL/slm_infer.py models/SpatialLM1.1-Qwen-0.5B $B/pred/slm_base_s$seed object $seed $B/lobby.ply $B/corridor.ply $B/campus.ply 2>&1 | grep -E "^OK|^voxel|Error|Traceback"
done
SLM_TILE=10 python /mnt/e/WSL/slm_infer.py models/SpatialLM1.1-Qwen-0.5B-ARKitScenes-SFT $B/pred/slm_arkit_s0 object 0 $B/lobby.ply $B/corridor.ply $B/campus.ply 2>&1 | grep -E "^OK|^voxel|Error|Traceback"
SLM_TILE=10 python /mnt/e/WSL/slm_infer.py models/SpatialLM1.1-Qwen-0.5B $B/pred/slm_base_prompted_s0 object 0 $B/lobby.ply $B/corridor.ply $B/campus.ply --cat chair sofa bench stool table plants cabinet 2>&1 | grep -E "^OK|^voxel|Error|Traceback"
echo "=== sonata"
pip install -q -e /opt/work/sonata --no-deps 2>&1 | tail -1
cd /mnt/e/WSL
python sonata_bench.py $B lobby corridor campus 2>&1 | grep -v -i warn | tail -12
mkdir -p $B/pred/sonata_zeroshot && cp $B/pred_sonata_zs/*.json $B/pred/sonata_zeroshot/
echo "=== finetune"
python finetune_head.py $B /mnt/e/WSL/gt 2>&1 | grep -v -i warn | tail -12
mkdir -p $B/pred/sonata_finetuned && cp $B/pred_sonata_ft/*.json $B/pred/sonata_finetuned/
mkdir -p $B/pred/our_pipeline && cp $B/pred_pipeline/*.json $B/pred/our_pipeline/
echo ALL_DONE
