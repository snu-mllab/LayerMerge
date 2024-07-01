import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import socket
import pandas as pd
import re
from itertools import product, combinations
from datetime import datetime
from typing import List, Tuple

from layer_merge.measure import torch_time
from layer_merge.models.resnet import ResNet34, ResNet50
from layer_merge.models.resnet_layer import LayerResNet
from layer_merge.models.mobilenetv2 import MobileNetV2
from layer_merge.models.mobilenetv2_layer import LayerMobileNetV2
from layer_merge.models.ddpm import DDPMModel
from layer_merge.models.ddpm_layer import LayerDDPMModel
from layer_merge.models.ddpm_datasets import dataset2config


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


def get_conv(convs, ind):
    for node_pos, (_, layer) in convs.items():
        if node_pos == ind:
            return layer


def generate_time_table(
    model: LayerResNet,
    indices: List[int],
    dir: str,
    tag: str = "",
):
    """
    This function computes the $T(\cdot)$ table
    """
    cnt = torch.cuda.device_count()
    env = f"{socket.gethostname()}_gpu{cnt}"
    out_shape = model.outs

    blks_dict = {
        "index": [],
        "time": [],
        "stdev": [],
    }
    print(indices)
    for index in indices:

        now = datetime.now().strftime("%m/%d %H:%M:%S")

        conv = get_conv(model.convs, index)

        print(f"index: {index:>3} | {now}")

        merged_model = nn.Sequential(conv, nn.ReLU(inplace=True))
        conv.bias = nn.Parameter(torch.randn(conv.out_channels))
        print(conv)
        print(out_shape[index - 1])

        time, std = torch_time(
            merged_model,
            out_shape[index - 1],
            f"index : {index}",
            True,
            rep=200,
            warmup=300,
        )
        blks_dict["index"].append(index)
        blks_dict["time"].append(time)
        blks_dict["stdev"].append(std)

        print()

    blks = pd.DataFrame(blks_dict)
    if tag:
        filename = f"time_{tag}.csv"
    else:
        filename = f"time_{env}.csv"
    os.makedirs(dir, exist_ok=True)
    csv_path = os.path.join(dir, filename)
    blks.to_csv(csv_path, sep=",", index=False)


def normalize_imp(file_path, arch):
    df = pd.read_csv(file_path)
    dir_name, file_name = os.path.split(file_path)
    output_file_path = os.path.join(dir_name, "normalized_" + file_name)

    # Calculate the 'importance' column
    if arch.startswith("ddpm"):
        df["importance"] = np.exp(df["val_loss_ratio"])
    else:
        df["importance"] = np.exp(-df["val_acc"] / 100)
    # df['importance'] = 1 + df['val_acc'] / 100

    # Selecting the required columns
    output_df = df[["id", "index", "importance"]]
    output_df.to_csv(output_file_path)


def read_csv_and_create_dict(path):
    df = pd.read_csv(path)
    if "importance" in df:
        items = df["importance"]
    elif "time" in df:
        items = df["time"]
    return dict(zip(df["index"], items))


def optimal_patterns(flt_time_limit, time_path, imp_path, prec):
    time_dict = read_csv_and_create_dict(time_path)
    imp_dict = read_csv_and_create_dict(imp_path)

    time_limit = int(flt_time_limit * prec)

    mandatory_time = 0
    conv_ind = set()
    for index, time in time_dict.items():
        if index not in imp_dict:
            mandatory_time += int(time * prec)
            conv_ind.add(index)

    # Check if mandatory activities exceed time limit
    if mandatory_time > time_limit:
        raise ValueError("Mandatory activities exceed time limit")

    # Adjust time limit for mandatory activities
    adjusted_time_limit = time_limit - mandatory_time

    dp_tab = dict()
    argmax_tab = dict()
    dp_tab[0] = {key: 0 for key in range(adjusted_time_limit + 1)}
    argmax_tab[0] = {key: set() for key in range(adjusted_time_limit + 1)}
    for index, time in time_dict.items():
        dp_tab[index] = {key: 0 for key in range(adjusted_time_limit + 1)}
        argmax_tab[index] = {key: set() for key in range(adjusted_time_limit + 1)}

    prv_lyr_ind = 0
    # Knapsack Optimization
    for lyr_ind, time in time_dict.items():
        if lyr_ind in imp_dict:
            importance = imp_dict[lyr_ind]
            time = int(time * prec)
            for t in range(adjusted_time_limit + 1):
                if (t - time + 1) % 1000 == 0:
                    now = datetime.now().strftime("%m/%d %H:%M:%S")
                    prg = f"layer = {lyr_ind:>2} : {t:>5} / {adjusted_time_limit:>5}       ||      {now}"
                    print(prg)
                if (
                    t < time
                    or dp_tab[prv_lyr_ind][t]
                    > dp_tab[prv_lyr_ind][t - time] + importance
                ):
                    dp_tab[lyr_ind][t] = dp_tab[prv_lyr_ind][t]
                    argmax_tab[lyr_ind][t] = argmax_tab[prv_lyr_ind][t].copy()
                else:
                    dp_tab[lyr_ind][t] = dp_tab[prv_lyr_ind][t - time] + importance
                    argmax_tab[lyr_ind][t] = set.union(
                        argmax_tab[prv_lyr_ind][t - time], {lyr_ind}
                    )
            prv_lyr_ind = lyr_ind

    max_imp = 0
    for lyr_ind in argmax_tab[prv_lyr_ind][adjusted_time_limit]:
        conv_ind.add(lyr_ind)
        max_imp += imp_dict[lyr_ind]

    assert round(dp_tab[prv_lyr_ind][adjusted_time_limit], 7) == round(max_imp, 7)

    # Calculate summary values
    imp_sum = sum(imp_dict[i] for i in conv_ind if i in imp_dict)
    int_time_sum = sum(int(time_dict[i] * prec) for i in conv_ind)
    flt_time_sum = sum(time_dict[i] for i in conv_ind)

    return conv_ind, imp_sum, int_time_sum, flt_time_sum


def main():
    parser = argparse.ArgumentParser(
        description="Script for dynamic programming algorithm"
    )
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        choices=[
            "resnet34",
            "resnet50",
            "mobilenetv2",
            "mobilenetv2_w1.4",
            "ddpm_cifar10",
            "ddpm_cifar10_pruned",
        ],
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="utils/table",
        help="directory name",
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default="",
        help="tag",
    )
    parser.add_argument("--mode", type=str, choices=["time", "normalize", "solve"])
    parser.add_argument(
        "--time_path", type=str, default="", help="Path to the normal time table"
    )
    parser.add_argument(
        "--imp_path", type=str, default="", help="Path to the normal importance table"
    )
    parser.add_argument("--prec", type=float, default=10, help="Precision in solving")
    parser.add_argument(
        "--time_limit", type=float, default=None, help="Time limit in solving"
    )
    parser.add_argument(
        "--pruned_path", type=str, default=None, help="Path to the pruned model"
    )

    args = parser.parse_args()

    if args.arch == "resnet34":
        model: LayerResNet = ResNet34()
        model.__class__ = LayerResNet
    elif args.arch == "resnet50":
        model: LayerResNet = ResNet50()
        model.__class__ = LayerResNet
    elif args.arch == "mobilenetv2":
        model: LayerMobileNetV2 = MobileNetV2()
        model.__class__ = LayerMobileNetV2
    elif args.arch == "mobilenetv2_w1.4":
        model: LayerMobileNetV2 = MobileNetV2(width_mult=1.4)
        model.__class__ = LayerMobileNetV2
    elif args.arch == "ddpm_cifar10":
        model: LayerDDPMModel = DDPMModel(dataset2config("cifar10"))
        model.__class__ = LayerDDPMModel
    elif args.arch == "ddpm_cifar10_pruned":
        assert args.pruned_path is not None
        # To handle torch.load() and load the pruned model
        import sys
        sys.path.insert(0, '../../Diff-Pruning/exp_code')
        print("Loading checkpoint {}".format(args.pruned_path))
        device = torch.device("cuda")
        states = torch.load(args.pruned_path, map_location=device)
        model = states[0].to(device)
        model: LayerDDPMModel = model
        model.__class__ = LayerDDPMModel
    else:
        raise NotImplementedError()

    model.morph()
    model.trace()

    if args.mode == "time":
        generate_time_table(model, list(range(1, model.depth)), args.dir, args.tag)
    elif args.mode == "normalize":
        normalize_imp(args.imp_path, args.arch)
    elif args.mode == "solve":
        args.checkpoint = os.path.join(args.dir, f"p{args.prec}_tl{args.time_limit}")

        config = "Solving DP\n"
        config += f"ARCH          : {args.arch}\n"
        config += f"IMP PATH      : {args.imp_path}\n"
        config += f"TIME PATH     : {args.time_path}\n"
        config += f"TIME LIMIT    : {args.time_limit}\n"
        config += f"PRECISION     : {args.prec}\n"

        config += f"\n---------------------------\n\n"
        print(config)

        conv_ind, imp_sum, int_time, flt_time = optimal_patterns(
            flt_time_limit=args.time_limit,
            time_path=args.time_path,
            imp_path=args.imp_path,
            prec=args.prec,
        )

        conv_ind = set.union({0}, conv_ind)

        result = "\n---------------------------\n"

        result += f"CONV POS      : {sorted(list(conv_ind))}\n"
        result += f"IMP SUM       : {imp_sum}\n"
        result += f"INT TIME SUM  : {int_time}\n"
        result += f"FLT TIME SUM  : {flt_time}\n"

        print(result)

        os.makedirs(args.checkpoint, exist_ok=True)

        with open(os.path.join(args.checkpoint, "result.log"), "w") as f:
            f.write(config + result)
            f.write("\n")

        torch.save(
            {
                "conv_ind": conv_ind,
                "imp_sum": imp_sum,
                "int_time_sum": int_time,
                "flt_time_sum": flt_time,
            },
            os.path.join(args.checkpoint, "checkpoint.pth"),
        )
    else:
        raise NotImplementedError()
