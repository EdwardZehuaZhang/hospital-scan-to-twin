source /opt/conda/etc/profile.d/conda.sh; conda activate slm
B=/mnt/e/WSL/bench
pip install -q matplotlib 2>&1 | tail -1
cd /mnt/e/WSL
python finetune_head.py $B /mnt/e/WSL/gt 2>&1 | grep -v -i warn | tail -12
mkdir -p $B/pred/sonata_finetuned && cp $B/pred_sonata_ft/*.json $B/pred/sonata_finetuned/
mkdir -p $B/pred/sonata_zeroshot && cp $B/pred_sonata_zs/*.json $B/pred/sonata_zeroshot/
cd /opt/work/SpatialLM
SLM_TILE=10 python /mnt/e/WSL/slm_infer.py models/SpatialLM1.1-Qwen-0.5B-ARKitScenes-SFT $B/pred/slm_arkit_cat_s0 object 0 $B/lobby.ply $B/corridor.ply $B/campus.ply --cat chair stool table sofa cabinet shelf 2>&1 | grep -E "^OK|Error|Traceback"
echo FT_DONE
