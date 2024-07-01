import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import torch.backends.cudnn as cudnn

from models.imagenet import (
    MobileNetV2,
    LearnMobileNetV2,
    DepShrinkMobileNetV2,
    VGG,
    LearnVGG,
)
from models.model_op import unroll_conv_params, adjust_with_bn
from utils.misc import load_checkpoint
from utils.logger import Logger
from utils.measure import torch_time, compile_and_time

from layer_merge.models.mobilenetv2_merged_layer import make_depth_layer_mobilenet_v2
from layer_merge.models.mobilenetv2_layer import make_layer_mobilenet_v2

vgg_cfgs = {
    # (n, inp, oup, isize)
    "19": [
        (2, 3, 64, 224),
        (2, 64, 128, 112),
        (4, 128, 256, 56),
        (4, 256, 512, 28),
        (4, 512, 512, 14),
    ]
}


def str2bool(v):
    """Cast string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load(path, arch, width, nclass, logger):
    if arch == "learn_mobilenet_v2":
        model = nn.DataParallel(
            LearnMobileNetV2(num_classes=nclass, width_mult=width, add_relu=True)
        )
    elif arch == "dep_shrink_mobilenet_v2":
        model = nn.DataParallel(
            DepShrinkMobileNetV2(num_classes=nclass, width_mult=width)
        )
    elif arch == "mobilenet_v2":
        model = nn.DataParallel(MobileNetV2(num_classes=nclass, width_mult=width))
    elif arch == "depth_layer_mobilenet_v2":
        state = torch.load(path)
        act_ind, conv_ind = state["act_ind"], state["conv_ind"]
        model = make_depth_layer_mobilenet_v2(
            act_ind, conv_ind, num_classes=nclass, width_mult=width
        )
        merged_model = model.merge()
        return merged_model
    elif arch == "layer_mobilenet_v2":
        state = torch.load(path)
        conv_ind = state["conv_ind"]
        model = make_layer_mobilenet_v2(
            conv_ind, num_classes=nclass, width_mult=width
        )
        merged_model = model.merge()
        return merged_model
    # VGG
    elif arch == "vgg19":
        model = nn.DataParallel(VGG(cfg=vgg_cfgs["19"], num_classes=nclass))
    elif arch == "learn_vgg19":
        model = nn.DataParallel(LearnVGG(cfg=vgg_cfgs["19"], num_classes=nclass))
    load_checkpoint(model, arch, path, logger=logger)
    return model


def main():

    print(torch.cuda.device_count())  # Check the num of gpu

    cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Inference Time with TensorRT")
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="mobilenet_v2",
        type=str,
        choices=[
            "mobilenet_v2",
            "learn_mobilenet_v2",
            "depth_layer_mobilenet_v2",
            "layer_mobilenet_v2",
            "dep_shrink_mobilenet_v2",
            "vgg19",
            "learn_vgg19",
        ],
        help="model architecture",
    )
    parser.add_argument(
        "-w",
        "--width-mult",
        type=float,
        default=1.0,
        help="MobileNet model width multiplier.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="dir to the checkpoint (default: checkpoints)",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        metavar="FILE",
        help="filename of the checkopint (default: checkpoint.pth)",
    )
    parser.add_argument(
        "--nclass",
        type=int,
        default=1000,
        choices=[10, 100, 1000],
        help="number of classes",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="log name",
    )
    parser.add_argument(
        "--trt",
        default=True,
        type=str2bool,
        help="Whether to measure in TensorRT or not",
    )
    parser.add_argument(
        "--cpu",
        default=False,
        type=str2bool,
        help="Whether to measure with CPU",
    )

    args = parser.parse_args()
    if args.cpu:
        assert not args.trt

    logger = Logger(os.path.join(args.checkpoint, f"time_{args.name}.log"))
    logger.comment(str(args))
    logger.comment("")

    gpu_name = torch.cuda.get_device_name(0)
    logger.comment(gpu_name)

    ckpt = os.path.join(args.checkpoint, args.filename)
    model = load(ckpt, args.arch, args.width_mult, args.nclass, logger)
    model = model.to("cuda")
    model.eval()

    # Fuse batchnorm layers...
    if args.arch in [
        "mobilenet_v2",
        "vgg19",
    ]:
        model.module.merge()

    print(model)

    logger.comment(f"TensorRT  :  {args.trt}")
    logger.comment(f"CPU             :  {args.cpu}")

    if args.arch.startswith("depth_layer") or args.arch.startswith("layer"):
        if args.trt:
            time, std = compile_and_time(
                model, (128, 3, 224, 224), args.arch, True, logger=logger
            )
        else:
            time, std = torch_time(model, (128, 3, 224, 224), args.arch, True, logger=logger)
    else:
        if args.cpu:
            time, std = model.module.cpu_time(txt=args.arch, verb=True)
        else:
            time, std = model.module.time(txt=args.arch, verb=True, trt=args.trt)
    logger.comment("")
    logger.comment("TIME MEAN")
    logger.comment(str([time]))
    logger.comment("STD")
    logger.comment(str([std]))


if __name__ == "__main__":
    main()
