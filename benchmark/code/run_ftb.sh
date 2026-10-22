source /opt/conda/etc/profile.d/conda.sh; conda activate slm
cd /mnt/e/WSL; B=/mnt/e/WSL/bench
BINARY=1 python finetune_head.py $B /mnt/e/WSL/gt 2>&1 | grep -v -i warn | tail -6
mkdir -p $B/pred/sonata_finetuned_binary && cp $B/pred_sonata_ft_binary/*.json $B/pred/sonata_finetuned_binary/
echo FTB_DONE
