python -B finetune.py \
--config celeba.yml \
--exp $2 \
--sample \
--timesteps 100 \
--eta 0 \
--ni \
--doc sample \
--skip_type uniform  \
--pruning_ratio 0.0 \
--fid \
--use_ema \
--restore_from $1 \