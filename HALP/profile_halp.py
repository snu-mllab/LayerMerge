# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
#
# Official PyTorch implementation of NeurIPS2022 paper
# Structural Pruning via Latency-Saliency Knapsack
# Maying Shen, Hongxu Yin, Pavlo Molchanov, Lei Mao, Jianna Liu and Jose M. Alvarez
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# --------------------------------------------------------

import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from models import get_model, fuse_model
from utils.model_summary import model_summary
from utils.utils import ExpConfig

# Torch TensorRT
from layer_merge.measure import compile_and_time

parser = argparse.ArgumentParser(description="Profiling the inference time.")
parser.add_argument(
    "--exp",
    type=str,
    default="configs/exp_configs/rn50_imagenet_baseline.yaml",
    help="Config file for the experiment.",
)
parser.add_argument(
    "--model_path", type=str, default=None, help="The path of the model."
)
parser.add_argument(
    "--mask_path", type=str, required=False, help="The path of the mask file."
)
parser.add_argument(
    "--depth_path",
    type=str,
    default=None,
    help="The path of the solution to the depth compression.",
)
parser.add_argument(
    "--depth_method",
    type=str,
    choices=["kim23efficient", "kim24layermerge", "kim24layer", None],
    default=None,
    help="The depth compression method",
)
parser.add_argument(
    "--batch_size", type=int, default=256, help="The batch size of inference."
)
parser.add_argument(
    "--trt", action="store_true", help="Compile and measure with TensorRT."
)
args = parser.parse_args()


def main():
    with open(args.exp) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    exp_cfg = ExpConfig(cfg)
    exp_cfg.override_config(vars(args))

    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.set_grad_enabled(False)
    gpu = 0
    cuda = True

    print(args.mask_path)
    model = get_model(
        exp_cfg.arch,
        exp_cfg.class_num,
        exp_cfg.enable_bias,
        group_mask_file=args.mask_path,
        depth_file=args.depth_path,
        depth_method=args.depth_method,
    )
    # if args.model_path is not None:
    #     resume_ckpt = torch.load(args.model_path, map_location="cpu")
    #     if "state_dict" in resume_ckpt:
    #         resume_ckpt_state_dict = resume_ckpt["state_dict"]
    #     else:
    #         resume_ckpt_state_dict = resume_ckpt
    #     model.load_state_dict(
    #         {k.replace("module.", ""): v for k, v in resume_ckpt_state_dict.items()}
    #     )

    if args.depth_path:
        state = torch.load(args.depth_path)
        if args.depth_method == "kim23efficient":
            merge_pos = state["merge_pos"]
            model = model.merge(merge_pos)
        elif args.depth_method == "kim24layermerge":
            merge_pos = state["act_ind"]
            model = model.merge(merge_pos)
        elif args.depth_method == "kim24layer":
            model = model.merge()
    else:
        model = fuse_model(exp_cfg.arch, model)
    print(model)

    device = torch.device(gpu)

    model.eval()
    model.to(device)

    if exp_cfg.dataset_name.lower() == "imagenet":
        input = torch.randn(exp_cfg.batch_size, 3, 224, 224)
    elif exp_cfg.dataset_name.lower() == "cifar10":
        input = torch.randn(exp_cfg.batch_size, 3, 32, 32)
    else:
        raise NotImplementedError

    if args.trt:
        if args.model_path:
            path_name = args.model_path
        elif args.mask_path:
            path_name = args.mask_path
        elif args.depth_path:
            path_name = args.depth_path
        else:
            path_name = exp_cfg.arch

        result, _ = compile_and_time(
            model, 
            tuple(input.shape),
            path_name,
            verb=False,
        )
        print("Infer time (ms/image)", result / exp_cfg.batch_size)
        print("FPS:", exp_cfg.batch_size * 1e3 / result)
    else:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        times = []
        for i in range(40):
            input = input.to(device)
            start_evt.record()
            output = model(input)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_time = start_evt.elapsed_time(end_evt)
            # warmup
            if i < 10:
                continue
            times.append(elapsed_time)
        print("Infer time (ms/image)", np.mean(times) / exp_cfg.batch_size)
        print("FPS:", exp_cfg.batch_size * 1e3 / np.mean(times))

    if exp_cfg.dataset_name.lower() == "imagenet":
        input = torch.randn(1, 3, 224, 224)
    elif exp_cfg.dataset_name.lower() == "cifar10":
        input = torch.randn(1, 3, 32, 32)
    else:
        raise NotImplementedError
    flops = model_summary(model, input.cuda())
    print("MACs(G): {:.3f}".format(flops / 1e9))


if __name__ == "__main__":
    main()
