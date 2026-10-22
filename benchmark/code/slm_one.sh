source /opt/conda/etc/profile.d/conda.sh; conda activate slm
cd /opt/work/SpatialLM
SLM_TILE=10 python /mnt/e/WSL/slm_infer.py models/SpatialLM1.1-Qwen-0.5B /mnt/e/WSL/bench/slm_try object 0 /mnt/e/WSL/bench/lobby.ply 2>&1 | grep -E "^OK|Traceback|Error"
head -4 /mnt/e/WSL/bench/slm_try/lobby.txt
echo ONE_DONE
