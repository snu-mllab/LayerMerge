python -B prune_ssim.py \
--config cifar10.yml \
--timesteps 100 \
--eta 0 \
--ni \
--doc post_training \
--skip_type quad  \
--pruning_ratio 0.2 \
--use_ema \
--use_pretrained \
--stage $1 \
--pruner "ours" \
--save_pruned_model run/pruned_v4/cifar10_pruned.pth \