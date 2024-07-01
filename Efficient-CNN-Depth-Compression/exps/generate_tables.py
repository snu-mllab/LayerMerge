import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import argparse
from itertools import product

from models.imagenet import LearnMobileNetV2, LearnVGG, vgg_cfgs
from models.imagenet import InvertedResidual, VGGBlock
from models.model_op import valid_blks
from utils.dp import (
    generate_time_table,
    generate_optimal_time_table,
    generate_ext_imp_table,
)
from utils.logger import Logger


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


def construct_ext_blks(arch, blks, blk_pos, exclude_zeros=True):
    ext_blks = []
    for st, end in blks:
        # Exclude blk_pos blocks that ends with zero
        if arch == "learn_mobilenet_v2":
            if exclude_zeros:
                if st == 0:
                    st_acts, end_acts = [1], [1]
                elif st in blk_pos:
                    st_acts, end_acts = [1, 0], [1]
                elif end in blk_pos:
                    st_acts, end_acts = [1], [1, 0]
                else:
                    st_acts, end_acts = [1], [1]
            else:
                st_acts = [1, 0] if st in blk_pos else [1]
                end_acts = [1, 0] if end in blk_pos else [1]
        elif arch == "learn_vgg19":
            st_acts, end_acts = [1], [1]
        else:
            raise NotImplementedError()
        for st_act, end_act in product(st_acts, end_acts):
            ext_blks.append((st, end, st_act, end_act))
    return ext_blks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for Generating Time Table")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="utils/table",
        help="directory name",
    )
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default="learn_mobilenet_v2",
        help="architecture of the network",
    )
    parser.add_argument(
        "-w",
        "--width-mult",
        type=float,
        default=None,
        help="width multiplier",
    )
    parser.add_argument(
        "--nclass",
        type=int,
        default=100,
        choices=[10, 100, 1000],
        help="number of classes",
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default="",
        help="tag",
    )
    parser.add_argument("--mode", type=str, choices=["time", "opt-time", "ext-imp"])
    parser.add_argument(
        "--time-path", type=str, default="", help="Path to the normal time table"
    )
    parser.add_argument(
        "--trt", type=str2bool, default=True, help="Path to the normal time table"
    )
    parser.add_argument(
        "--imp-path", type=str, default="", help="Path to the normal importance table"
    )
    parser.add_argument(
        "--score", type=str, default="", help="Score for the importance"
    )
    parser.add_argument(
        "--norm", type=str, default="default", help="Normalization method of score"
    )
    parser.add_argument("--alph", type=float, default=1.0, help="Alpha in normalizing")

    args = parser.parse_args()

    if args.mode == "time":
        logger = Logger(os.path.join(args.dir, f"time_table_{args.tag}.log"))

    assert args.arch in [
        "learn_mobilenet_v2",
        "learn_vgg19",
    ]
    if args.arch == "learn_mobilenet_v2":
        model = LearnMobileNetV2(
            num_classes=args.nclass, width_mult=args.width_mult, add_relu=True
        )
        blk_type = InvertedResidual
        act_num = model.get_act_info()[1]
        default = set(range(1, act_num + 1)) - set(range(2, act_num + 1, 3))
    elif args.arch == "learn_vgg19":
        model = LearnVGG(vgg_cfgs["19"], num_classes=args.nclass)
        act_num = model.get_act_info()[1]
        blk_type = VGGBlock
        default = set(range(1, act_num + 1))
    blks = valid_blks(model)

    if args.mode == "time":
        generate_time_table(
            model, blks, args.arch, blk_type, args.dir, args.tag, logger, args.trt
        )
    elif args.mode == "opt-time":
        assert args.time_path
        generate_optimal_time_table(blks, args.dir, args.tag, args.time_path)
    elif args.mode == "ext-imp":
        assert str(args.width_mult) in args.imp_path or args.width_mult == None
        blk_pos = list(model.get_blk_info()[0].values())
        ext_blks = construct_ext_blks(args.arch, blks, blk_pos)
        assert args.imp_path and args.score in ["train_acc", "val_acc"]
        generate_ext_imp_table(
            ext_blks,
            act_num,
            args.dir,
            args.imp_path,
            args.score,
            default,
            args.norm,
            args.alph,
        )
    else:
        raise NotImplementedError()
