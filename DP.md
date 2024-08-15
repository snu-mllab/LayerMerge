# Constructing look-up tables & Solving DP algorithm

We provide the instruction that can construct the look-up tables for latency and importance here. 
The latency measurements are obtained using a single RTX2080 Ti GPU, while the importance are measured using two RTX3090 GPUs.


### ResNet-34
1. Move to `HALP` directory.
    ```
    cd HALP
    ```
2. Construct the latency look-up table (we used single RTX2080 Ti GPU in the paper).
    ```
    lymg_kim24_dp -a resnet34 --mode time \
    -d LUT_kim24/time/rn34/ \
    -t fish_1206_batch128
    ```
3. Measure the importance look-up table entries (we used two RTX3090 GPU per entry). This process can be parallelized across multiple GPUs by setting `--from_blk` and `--to_blk` for each device.
    ```
    lymg_kim24_imp -a resnet34 \
    --pretrained pretrained/resnet34_full.pth \
    --save_path LUT_kim24/imp/rn34/par \
    --data /ssd_data/imagenet/ --from_blk 0 --to_blk 150
    ```
4. Aggregate importance table entries into a single file.
    ```
    lymg_agg -d LUT_kim24/imp/rn34 -n 150
    ```
5. Normalize the importance value.
    ```
    lymg_kim24_dp -a resnet34 --mode normalize \
    --imp_path LUT_kim24/imp/rn34/importance.csv
    ```
6. To solve DP problem with $T_0$ time budget, run
    ```
    lymg_kim24_dp -a resnet34 --mode solve \
    -d LUT_kim24/solve/rtx2080/rn34/ \
    --time_path LUT_kim24/time/rn34/time_fish_1206_batch128.csv \
    --imp_path LUT_kim24/imp/rn34/normalized_importance.csv \
    --time_limit {T_0}
    ```
    - To reproduce the numbers in Table 1, try `{T_0}` value from {`146.0`, `123.0`, `103.0`} and proceed to next step.

### MobileNetV2-(1.0/1.4)
1. Move to `Efficient-CNN-Depth-Compression` directory.
    ```
    cd Efficient-CNN-Depth-Compression
    ```
2. Construct the latency look-up table (we used single RTX2080 Ti GPU in the paper).
    ```bash
    # MobileNetV2-1.0
    lymg_kim24_dp -a mobilenetv2 --mode time \
    -d LUT_kim24/time/mbv2/ \
    -t fish_1211_batch128
    ```
    ```bash
    # MobileNetV2-1.4
    lymg_kim24_dp -a mobilenetv2_w1.4 --mode time \
    -d LUT_kim24/time/mbv2_w1.4/ \
    -t fish_0112_batch128
    ```
3. Measure the importance look-up table entries (we used two RTX3090 GPU per entry). This process can be parallelized across multiple GPUs by setting `--from_blk` and `--to_blk` for each device.
    ```bash
    # MobileNetV2-1.0
    lymg_kim24_imp -a mobilenetv2 \
    --pretrained pretrained/mobilenetv2_100_ra-b33bc2c4.pth \
    --save_path LUT_kim24/imp/mbv2/par \
    --data /ssd_data/imagenet/ --from_blk 0 --to_blk 391
    ```
    ```bash
    # MobileNetV2-1.4
    lymg_kim24_imp -a mobilenetv2_w1.4 \
    --pretrained pretrained/mobilenetv2_140_ra-21a4e913.pth \
    --save_path LUT_kim24/imp/mbv2_w1.4/par \
    --data /ssd_data/imagenet/ --from_blk 0 --to_blk 391
    ```
4. Aggregate importance table entries into a single file.
    ```bash
    # MobileNetV2-1.0
    lymg_agg -d LUT_kim24/imp/mbv2 -n 391
    ```
    ```bash
    # MobileNetV2-1.4
    lymg_agg -d LUT_kim24/imp/mbv2_w1.4 -n 391
    ```
5. Normalize the importance value.
    ```bash
    # MobileNetV2-1.0
    lymg_kim24_dp -a mobilenetv2 --mode normalize \
    --imp_path LUT_kim24/imp/mbv2/importance.csv
    ```
    ```bash
    # MobileNetV2-1.4
    lymg_kim24_dp -a mobilenetv2_w1.4 --mode normalize \
    --imp_path LUT_kim24/imp/mbv2_w1.4/importance.csv
    ```
6. To solve DP problem with $T_0$ time budget, run
    ```bash
    # MobileNetV2-1.0
    lymg_kim24_dp -a mobilenetv2 --mode solve \
    -d LUT_kim24/solve/rtx2080/mbv2/ \
    --time_path LUT_kim24/time/mbv2/time_fish_1211_batch128.csv \
    --imp_path LUT_kim24/imp/mbv2/normalized_importance.csv \
    --time_limit {T_0}
    ```
    - To reproduce the numbers in Table 2, try `{T_0}` value from {`22.3`, `18.5`, `15.6`, `13.4`} and proceed to next step.
    ```bash
    # MobileNetV2-1.4
    lymg_kim24_dp -a mobilenetv2_w1.4 --mode solve \
    -d LUT_kim24/solve/rtx2080/mbv2_w1.4/ \
    --time_path LUT_kim24/time/mbv2_w1.4/time_fish_0112_batch128.csv \
    --imp_path LUT_kim24/imp/mbv2_w1.4/normalized_importance.csv \
    --time_limit {T_0}
    ```
    - To reproduce the numbers in Table 3, try `{T_0}` value from {`26.1`, `25.0`, `21.0`, `18.0`} and proceed to next step.

### DDPM
1. Move to `Diff-Pruning/exp_code` directory.
    ```
    cd Diff-Pruning/exp_code
    ```
2. Construct the latency look-up table (we used single RTX2080 Ti GPU in the paper).
    ```
    lymg_kim24_dp -a ddpm_cifar10 --mode time
    -d LUT_kim24/time/ddpm_cifar10 \
    -t fish_0116_batch128 \
    ```
3. Measure the importance look-up table entries (we used two RTX3090 GPU per entry). This process can be parallelized across multiple GPUs by setting `--from_blk` and `--to_blk` for each device.
    ```
    slmg_kim24_imp -a ddpm_cifar10 \
    --imp_epoch 50 \
    --pretrained run/cache/diffusion_models_converted/ema_diffusion_cifar10_model/model-790000.ckpt \
    --save_path LUT_kim24/imp/ddpm_cifar10/par \
    --data /data_large/readonly/ --from_blk 0 --to_blk 98
    ```
4. Aggregate importance table entries into a single file.
    ```
    lymg_agg -d LUT_kim24/imp/ddpm_cifar10 -n 98
    ```
5. Normalize the importance value.
    ```
    lymg_kim24_dp -a ddpm_cifar10 --mode normalize \
    --imp_path LUT_kim24/imp/ddpm_cifar10/importance.csv
    ```
6. To solve DP problem with $T_0$ time budget, run
    ```
    lymg_kim24_dp -a ddpm_cifar10 --mode solve \
    -d LUT_kim24/solve/rtx2080/ddpm_cifar10/ \
    --time_path LUT_kim24/time/ddpm_cifar10/time_fish_0116_batch128.csv \
    --imp_path LUT_kim24/imp/ddpm_cifar10/normalized_importance.csv \
    --time_limit {T_0}
    ```
    - To reproduce the numbers in Table 4, try `{T_0}` value from {`48.0`, `46.0`, `38.0`} and proceed to next step.