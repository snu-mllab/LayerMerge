# Evaluating performance & latency speed-up

We provide the instruction that can evaluate the perfomance and latency speed-up here.
Ensure that checkpoints are placed in the correct paths as described in [README.md](README.md).

### ResNet-34
1. Move to `HALP` directory.
    ```
    cd HALP
    ```
2. To evaluate the performance,
    - For pre-trained network, run
        ```
        python multiproc.py --nproc_per_node 1 main.py \
        --data_root {IMAGENET_DIR} --eval_only \
        --exp configs/exp_configs/rn34_imagenet_baseline_eval.yaml \
        --pretrained pretrained/resnet34_full.pth
        ```
    - For compressed network ($T_0$ time budget), run
        ```
        python multiproc.py --nproc_per_node 1 main.py \
        --data_root {IMAGENET_DIR} --eval_only \
        --exp configs/exp_configs/rn34_imagenet_baseline_eval.yaml \
        --pretrained output_rtx2080/rn34_kim24layermerge_tl{T_0}/epoch_89.pth \
        --depth_path LUT_kim24/solve/rtx2080/rn34/p10_tl{T_0}/checkpoint.pth \
        --depth_method kim24layermerge
        ```
    - Replace `{IMAGENET_DIR}` with imagenet dataset directory.
    - Replace `{T_0}` with your desired time budget.

3. To profile the latency, 
    - For pre-trained network, run
        ```
        python profile_halp.py \
        --exp configs/exp_configs/rn34_imagenet_baseline_eval.yaml \
        --model_path pretrained/resnet34_full.pth
        ```
    - For compressed network ($T_0$ time budget), run
        ```
        python profile_halp.py \
        --exp configs/exp_configs/rn34_imagenet_baseline_eval.yaml \
        --depth_path LUT_kim24/solve/rtx2080/rn34/p10_tl{T_0}/checkpoint.pth \
        --depth_method kim24layermerge
        ```
    - Replace `{T_0}` with your desired time budget.

### MobileNetV2-(1.0/1.4)
1. Move to `Efficient-CNN-Depth-Compression` directory.
    ```
    cd Efficient-CNN-Depth-Compression
    ```
2. To evaluate the performance,
    - For pre-trained network, run
        ```bash
        # MobileNetV2-1.0
        python exps/main.py -a mobilenet_v2 \
        -d {$IMAGENET_DIR} -m eval --width-mult 1.0 \
        -c pretrained/ -f mobilenetv2_100_ra-b33bc2c4.pth
        ```
        ```bash
        # MobileNetV2-1.4
        python exps/main.py -a mobilenet_v2 \
        -d {$IMAGENET_DIR} -m eval --width-mult 1.4 \
        -c pretrained/ -f mobilenetv2_140_ra-21a4e913.pth
        ```
    - For compressed network ($T_0$ time budget), run
        ```bash
        # MobileNetV2-1.0
        python exps/main.py -a depth_layer_mobilenet_v2 \
        -d {IMAGENET_DIR} -m eval --width-mult 1.0 \
        -c output_rtx2080/p10_tl{T_0} -f checkpoint_ft_lr0.05.pth \
        --act-path LUT_kim24/solve/rtx2080/mbv2/p10_tl{T_0}/checkpoint.pth
        ```
        ```bash
        # MobileNetV2-1.4
        python exps/main.py -a depth_layer_mobilenet_v2 \
        -d {IMAGENET_DIR} -m eval --width-mult 1.4 \
        -c output_w1.4_rtx2080/p10_tl{T_0}_aug -f checkpoint_ft_lr0.1.pth \
        --act-path LUT_kim24/solve/rtx2080/mbv2/p10_tl{T_0}/checkpoint.pth
        ```
    - Replace `{IMAGENET_DIR}` with imagenet dataset directory.
    - Replace `{T_0}` with your desired time budget.
3. To profile the latency (PyTorch), 
    - For pre-trained network, run
        ```bash
        # MobileNetV2-1.0
        python exps/inference_trt.py -a mobilenet_v2 --width-mult 1.0 \
        -c pretrained/ -f mobilenetv2_100_ra-b33bc2c4.pth --trt False 
        ```
        ```bash
        # MobileNetV2-1.4
        python exps/inference_trt.py -a mobilenet_v2 --width-mult 1.4 \
        -c pretrained/ -f mobilenetv2_140_ra-21a4e913.pth --trt False
        ```
    - For compressed network ($T_0$ time budget), run
        ```bash
        # MobileNetV2-1.0
        python exps/inference_trt.py -a depth_layer_mobilenet_v2 --width-mult 1.0 \
        -c LUT_kim24/solve/rtx2080/mbv2/p10_tl{T_0} -f checkpoint.pth --trt False
        ```
        ```bash
        # MobileNetV2-1.4
        python exps/inference_trt.py -a depth_layer_mobilenet_v2 --width-mult 1.4 \
        -c LUT_kim24/solve/rtx2080/mbv2_w1.4/p10_tl{T_0} -f checkpoint.pth --trt False
        ```
    - To profile TensorRT latency, give `--trt True` option in the above commands.
    - Replace `{T_0}` with your desired time budget.

### DDPM
1. Move to `Diff-Pruning/exp_code` directory and extract the statistics of the data.
    ```
    cd Diff-Pruning/exp_code
    python tools/extract_cifar10.py --output data
    python fid_score.py --save-stats data/cifar10 run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256
    ```
2. To sample from the model,
    - For pre-trained model, run
        ```bash
        python finetune.py --sample --fid --config cifar10.yml --timesteps 100 --eta 0 --ni \
        --exp run/sample_pretrained \
        --doc sample --skip_type quad --use_ema --use_pretrained
        ```
    - For compressed network ($T_0$ time budget), run
        ```bash
        python finetune.py --sample --fid --config cifar10.yml --timesteps 100 --eta 0 --ni \
        --exp run/sample_depth_layer/output_rtx2080/ddpm_cifar10/p10_tl{T_0}/ \
        --doc sample --skip_type quad --use_ema \
        --restore_from run/output_rtx2080/ddpm_cifar10/p10_tl{T_0}/logs/post_training/ckpt_100000.pth \
        --depth_path LUT_kim24/solve/rtx2080/ddpm_cifar10/p10_tl{T_0}/checkpoint.pth \
        --depth_method kim24layermerge
        ```
    - Replace `{T_0}` with your desired time budget.
3. To evaluate FID score,
    - For pre-trained model, run
        ```bash
        python fid_score.py run/sample_pretrained run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256
        ```
    - For compressed network ($T_0$ time budget), run
        ```bash
        python fid_score.py run/sample_depth_layer/output_rtx2080/ddpm_cifar10/p10_tl{T_0}/ \
        run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256
        ```
    - Replace `{T_0}` with your desired time budget.
4. To profile the latency,
    - For pre-trained model, run
        ```bash
        python finetune.py --measure --fid --config cifar10.yml --timesteps 100 --eta 0 --ni \
        --exp run/time_pretrained \
        --doc sample --skip_type quad --use_ema --use_pretrained
        ```
    - For compressed network ($T_0$ time budget), run
        ```bash
        python finetune.py --measure --fid --config cifar10.yml --timesteps 100 --eta 0 --ni \
        --exp run/sample_depth_layer/output_rtx2080/ddpm_cifar10/p10_tl{T_0}/ \
        --doc sample --skip_type quad --use_ema \
        --restore_from run/output_rtx2080/ddpm_cifar10/p10_tl{T_0}/logs/post_training/ckpt_100000.pth \
        --depth_path LUT_kim24/solve/rtx2080/ddpm_cifar10/p10_tl{T_0}/checkpoint.pth \
        --depth_method kim24layermerge
        ```
    - Replace `{T_0}` with your desired time budget.
