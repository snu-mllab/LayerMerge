python -B prune_test.py \
--config cifar10.yml \
--timesteps 100 \
--eta 0 \
--ni \
--doc post_training \
--skip_type quad  \
--pruning_ratio 0.3 \
--use_ema \
--use_pretrained \
--pruner "$1" \
--save_pruned_model run/pruned_test/cifar10_pruned_$1_0.2.pth \