python -B finetune.py \
--config cifar10.yml \
--timesteps 100 \
--eta 0 \
--ni \
--exp run/finetune/cifar10_pruned_taylor_0.3_real_x_finetuned \
--doc post_training \
--skip_type quad  \
--pruning_ratio 0.3 \
--use_ema \
--use_pretrained \
--pruner taylor \
--load_pruned_model run/pruned/cifar10_pruned_taylor_0.3_real_x.pth \