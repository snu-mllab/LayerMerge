python -B finetune.py \
--config cifar10.yml \
--exp "$2" \
--measure \
--timesteps 100 \
--eta 0 \
--ni \
--doc sample \
--skip_type quad  \
--pruning_ratio 0.0 \
--fid \
--use_ema \
--restore_from "$1" \