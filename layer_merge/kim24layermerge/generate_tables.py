import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import socket
import pandas as pd
import re
from itertools import product, combinations
from collections import Counter
from datetime import datetime
from typing import List, Tuple

from layer_merge.measure import torch_time
from layer_merge.models.resnet import ResNet34, ResNet50
from layer_merge.models.resnet_merged_layer import DepthLayerResNet
from layer_merge.models.mobilenetv2 import MobileNetV2
from layer_merge.models.mobilenetv2_merged_layer import DepthLayerMobileNetV2
from layer_merge.models.ddpm import DDPMModel
from layer_merge.models.ddpm_merged_layer import DepthLayerDDPMModel
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


def read_csv_and_create_dict(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Create the dictionary
    result_dict = {}
    for index, row in df.iterrows():
        key = f"st{row['st']}_end{row['end']}"
        childkey = f"ker{row['ker']}_str{row['str']}_is_dw_{row['is_dw']}"

        if not key in result_dict:
            result_dict[key] = dict()

        if "importance" in row:
            result_dict[key][childkey] = (row["importance"], row["conv_comb"])
        elif "time" in row:
            result_dict[key][childkey] = row["time"]
        else:
            raise NotImplementedError()

    return result_dict


def valid_blks(model):
    skip_s2t = model.skip_s2t
    if hasattr(model, "str_pos"):
        str_pos = sorted(list(model.str_pos))
    elif hasattr(model, "attn_pos"):
        skip_src, skip_tgt = set(model.skip_s2t.keys()), set(model.skip_s2t.values())
        str_pos = sorted(list(set.union(skip_src, skip_tgt, model.attn_pos)))
    else:
        raise NotImplementedError()
    act_num = model.depth - 1
    breaks = str_pos + [act_num]
    phase = 0
    b_pos = breaks[phase]
    skip_pos = b_pos
    blks = []
    for st in range(0, act_num):
        if st == b_pos:
            if b_pos == breaks[phase]:
                phase += 1
            b_pos = breaks[phase]
        if st == skip_pos:
            skip_pos = b_pos
        end = st + 1
        while end <= min(b_pos, skip_pos):
            blks.append((st, end))
            if end in skip_s2t:
                end = skip_s2t[end]
            else:
                end += 1
        if st in skip_s2t:
            skip_pos = skip_s2t[st]
    return blks


def get_conv_lst(convs, st, end):
    lst = []
    assert st < end
    for node_pos, (_, layer) in convs.items():
        if node_pos > st and node_pos <= end:
            lst.append(layer)
        if node_pos == end:
            return lst


def get_possible_merged_conv(conv_num, include_identity=False):
    """
    Return the possible configuration of the merged convolutions that can appear with the given config of convs.
    Also, return the corresponding combination that makes the merged convolution

    Parameters:
    conv_num (dict): Dict of {(kernel_size, is_dw): [min_num, max_num]}.

    Returns:
    results : List of [(merged_kernel_size, merged_is_dw)].
    reulsts2comb : Dict of {(merged_kernel_size, merged_is_dw): Dict of {(kernel, is_dw): corresponding_num}}.
    """
    # Generate all combinations of val_i within the ranges of min_val_i and max_val_i
    ranges = (range(min_val, max_val + 1) for min_val, max_val in conv_num.values())
    combinations = list(product(*ranges))

    # Initialize the list to store results
    results = []
    result2comb = dict()

    for combo in combinations:
        is_identity = False
        # Calculate the sums
        sum_ki_vali = sum(
            kernel_size * val for (kernel_size, _), val in zip(conv_num.keys(), combo)
        )
        sum_vali = sum(combo)
        if sum_vali == 0:
            if not include_identity:
                continue
            is_identity = True

        # Calculate the is_depthwise flag for the combination
        is_depthwise_combination = True
        for (_, is_depthwise), val in zip(conv_num.keys(), combo):
            # If we find a non-zero value for a key where is_depthwise is False, set the flag to False
            if not is_depthwise and val != 0:
                is_depthwise_combination = False
                break

        # Calculate the result for the combination
        if is_identity:
            # If the combination is the identity, denote it by kernel size 0
            result = 0
        else:
            result = sum_ki_vali - sum_vali + 1
        results.append((result, is_depthwise_combination))

        result2comb[(result, is_depthwise_combination)] = dict(
            zip(conv_num.keys(), combo)
        )
    return results, result2comb


def simulate_list_merge_layer_prune(convs: List[nn.Conv2d], include_identity=False):
    """
    Find the possible merged convolution with the corresponding convolution indices.

    Parameters:
    convs (list of nn.Module): List of convolutional layers.

    Returns:
    merged_convs : Dict of {key (str): merged_conv (nn.Module)}.
    merged_comb : Dict of {key (str): corresponding conv indices that constitues merged conv (tuple)}.
    """
    conv_max = dict()
    conv_min = dict()

    new_stride = 1
    for x in convs:
        assert isinstance(x, nn.Conv2d)
        k = x.kernel_size[0]
        is_dw = x.groups == x.out_channels

        if (k, is_dw) in conv_max:
            conv_max[(k, is_dw)] += 1
        else:
            conv_max[(k, is_dw)] = 1

        if not (k, is_dw) in conv_min:
            conv_min[(k, is_dw)] = 0

        # Irreducible convolutions
        if x.in_channels != x.out_channels or x.stride[0] > 1:
            conv_min[(k, is_dw)] += 1

        if new_stride > 1:
            # Allow only the kernel-1 conv follows after the stride-2 conv.
            assert k == 1
        if x.stride[0] > 1:
            new_stride = x.stride[0]

    conv_num = dict()
    for (k, is_dw) in conv_max:
        conv_num[(k, is_dw)] = conv_min[(k, is_dw)], conv_max[(k, is_dw)]

    results, result2comb = get_possible_merged_conv(conv_num, include_identity)

    new_in_channels = convs[0].in_channels
    new_out_channels = convs[-1].out_channels

    merged_convs, merged_comb = dict(), dict()
    for new_kernel, new_is_dw in results:
        new_pad = new_kernel // 2
        if new_kernel > 0:
            new_conv = nn.Conv2d(
                new_in_channels,
                new_out_channels,
                new_kernel,
                new_stride,
                new_pad,
                groups=new_out_channels if new_is_dw else 1,
                bias=True,
            )
        else:
            new_conv = nn.Identity()
        key = f"in_{new_in_channels}_out_{new_out_channels}_ker_{new_kernel}_str_{new_stride}_pad_{new_pad}_dw_{new_is_dw}"

        merged_convs[key] = new_conv
        merged_comb[key] = result2comb[(new_kernel, new_is_dw)]
    return merged_convs, merged_comb

def cmp_ignore_zeros(dict1, dict2):
    # Removing entries with value 0 from both dictionaries
    filtered_dict1 = {k: v for k, v in dict1.items() if v != 0}
    filtered_dict2 = {k: v for k, v in dict2.items() if v != 0}

    # Comparing the filtered dictionaries
    return filtered_dict1 == filtered_dict2

def find_candidates_by_comb(conv, comb, offset, irreducible=None):
    """
    Find candidate indices with an offset such that only these indices will meet the specified criteria in 'comb',
    including any specified irreducible indices.

    Parameters:
    conv (list of nn.Modules): List of convolutional layers.
    comb (dict): Dictionary specifying the desired number of convolutional layers with specific characteristics.
                 The key is a tuple (kernel_size, is_depthwise), and the value is the desired count of such layers.
    offset (int): Offset value to be added to each index in the candidate tuples.
    irreducible (set, optional): Set of indices of irreducible layers. Candidates will include these indices.

    Returns:
    list of tuples: Each tuple represents candidate indices (with offset) that meet the criteria specified in 'comb',
                    and includes the irreducible indices, if provided.
    """
    candidates = []

    # Adjust the range for combinations to ensure irreducible indices are included
    min_range = len(irreducible) if irreducible else 1

    for i in range(min_range, len(conv) + 1):
        for indices in combinations(range(len(conv)), i):
            layer_indices = tuple(idx + offset for idx in indices)
            # Skip combinations that don't include all irreducible indices
            if irreducible and not irreducible.issubset(layer_indices):
                continue

            layer_count = {}
            for idx in indices:
                layer = conv[idx]
                # Check if layer is a Conv2d and get its kernel size
                if isinstance(layer, nn.Conv2d):
                    ker = (
                        layer.kernel_size[0]
                        if isinstance(layer.kernel_size, tuple)
                        else layer.kernel_size
                    )
                    is_dw = (
                        layer.in_channels == layer.groups
                    )  # Check if layer is depthwise
                    key = (ker, is_dw)
                    layer_count[key] = layer_count.get(key, 0) + 1

            # Check if the current set of indices matches the comb criteria
            if cmp_ignore_zeros(layer_count, comb):
                candidates.append(layer_indices)

    return candidates


def generate_time_table(
    model: DepthLayerResNet,
    blks: List[Tuple[int, int]],
    dir: str,
    tag: str = "",
):
    """
    This function computes the $T(\cdot, \cdot)$ table in the paper
    """
    cnt = torch.cuda.device_count()
    env = f"{socket.gethostname()}_gpu{cnt}"
    out_shape = model.outs

    blks = valid_blks(model)
    blks_dict = {
        "id": [],
        "st": [],
        "end": [],
        "ker": [],
        "str": [],
        "is_dw": [],
        "time": [],
        "stdev": [],
    }
    ind = 0

    if hasattr(model, "upsample_pos"):
        upsample_blks = [(upos - 1, upos) for upos in model.upsample_pos]
    else:
        upsample_blks = []

    for (st, end) in blks:

        now = datetime.now().strftime("%m/%d %H:%M:%S")

        conv_lst = get_conv_lst(model.convs, st, end)

        print("------")
        print("st:", st, "end:", end)
        print("conv_lst", conv_lst)
        print("------")
        print()
        is_upsample = (st, end) in upsample_blks
        merged_convs, _ = simulate_list_merge_layer_prune(
            conv_lst, include_identity=is_upsample
        )
        for key in merged_convs:
            print(
                f"ind: {ind:>3} | st: {st:>2} | end: {end:>2} | key: {key:>2} | {now}"
            )

            # Extract numbers after each parameter prefix
            params = re.findall(
                r"in_(\d+)_out_(\d+)_ker_(\d+)_str_(\d+)_pad_(\d+)_dw_(\w+)", key
            )

            # If a match is found, create a dictionary with the parameter values
            if params:
                param_names = ["in_ch", "out_ch", "ker", "str", "pad", "is_dw"]
                params_dict = dict(zip(param_names, params[0]))
                for pkey in params_dict:
                    if pkey == "is_dw":
                        params_dict[pkey] = str2bool(params_dict[pkey])
                    else:
                        params_dict[pkey] = int(params_dict[pkey])
            else:
                print("No match found. Check the input string format.")

            merged_model = nn.Sequential(merged_convs[key], nn.ReLU(inplace=True))
            print(merged_convs[key])

            if isinstance(merged_convs[key], nn.Identity):
                time, std = 0.0, 0.0
            else:
                time, std = torch_time(
                    merged_model,
                    out_shape[st],
                    f"st : {st:>2} | end : {end:>2}",
                    True,
                    rep=200,
                    warmup=300,
                )
            blks_dict["id"].append(ind)
            blks_dict["st"].append(st)
            blks_dict["end"].append(end)
            blks_dict["ker"].append(params_dict["ker"])
            blks_dict["str"].append(params_dict["str"])
            blks_dict["is_dw"].append(params_dict["is_dw"])
            blks_dict["time"].append(time)
            blks_dict["stdev"].append(std)

            ind += 1
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
        df["importance"] = np.exp(-df["val_loss_ratio"])
    else:
        df["importance"] = np.exp(df["val_acc"] / 100)
    # df['importance'] = 1 + df['val_acc'] / 100

    # Selecting the required columns
    output_df = df[
        ["id", "st", "end", "ker", "str", "is_dw", "conv_comb", "importance"]
    ]
    output_df.to_csv(output_file_path)


def optimal_patterns(
    flt_time_limit: float,
    act_num: int,
    time_path: str,
    imp_path: str,
    prec: float = 100,
    verbose: bool = False,
):

    # key format is f"st{st}_end{end}_ker{ker}_str{str}_is_dw_{is_dw}".
    time_dict = read_csv_and_create_dict(time_path)
    imp_dict = read_csv_and_create_dict(imp_path)

    time_limit = int(flt_time_limit * prec)

    # Sum of maximum importance
    dp_tab = dict()
    # Sum of optimal time
    dp_time_tab = dict()
    # Initialization
    for t in range(0, time_limit):
        dp_tab[(t, 0)] = 0
        dp_time_tab[(t, 0)] = 0
    max_tab = dict()
    children_tab = dict()
    for end in range(1, act_num + 1):
        t_min = float("Inf")
        for st in range(end):
            key = f"st{st}_end{end}"
            if not key in time_dict:
                continue
            for g_flt in time_dict[f"st{st}_end{end}"].values():
                cand = int(g_flt * prec)
                if t_min > cand:
                    t_min = cand

        # Impossible time limit
        if t_min > time_limit:
            print(f"Impossible time limit : {time_limit}")
            exit()

        for t in range(t_min, time_limit + 1):
            if verbose and (t - t_min) % 1000 == 0:
                now = datetime.now().strftime("%m/%d %H:%M:%S")
                prg = f"end = {end:>2} : {t - t_min:>5} / {time_limit - t_min:>5}       ||      {now}"
                print(prg)

            max_imp = float("-Inf")
            for st in range(end):
                key = f"st{st}_end{end}"
                if not key in time_dict:
                    continue
                for (childkey, g_flt), (h, h_comb) in zip(
                    time_dict[key].items(), imp_dict[key].values()
                ):
                    g = int(g_flt * prec)
                    # Skip for impossible combination
                    if not (t - g, st) in dp_tab:
                        continue
                    cand = dp_tab[(t - g, st)] + h
                    cand_time = dp_time_tab[(t - g, st)] + g

                    # For numerical stability of importance value
                    # (we round the values before comparison between float)
                    r_imp, r_cand = round(max_imp, 7), round(cand, 7)
                    # Choose maximum imp (on the tie we choose the one with faster time)
                    if r_imp < r_cand or (r_imp == r_cand and cand_time < opt_time):
                        max_imp = cand
                        opt_time = cand_time
                        argmax_imp = (t - g, st)
                        if st == 0:
                            children = {end: (childkey, h_comb)}
                        else:
                            children = children_tab[(t - g, st)].copy()
                            children[end] = (childkey, h_comb)

            # Skip for impossible combination (or combination that we do not take into account)
            if max_imp != float("-Inf"):
                dp_tab[(t, end)] = max_imp
                dp_time_tab[(t, end)] = opt_time
                max_tab[(t, end)] = argmax_imp
                children_tab[(t, end)] = children

    # t, end = time_limit, act_num
    if not (t, end) in dp_tab:
        print(f"Impossible time limit : {t}")
        exit()

    children_dict = children_tab[(t, end)]
    act_ind, conv_ind = {end}, set()
    while end > 0:
        act_ind = set.union(act_ind, {end})
        t, end = max_tab[(t, end)]

    assert dp_tab[(t, end)] == 0

    st = 0
    imp_sum, int_time_sum, flt_time_sum = 0, 0, 0
    for pos in sorted(list(act_ind)):
        if pos == 0:
            continue
        child_key, child_comb = children_dict[pos]
        imp, _ = imp_dict[f"st{st}_end{pos}"][child_key]
        flt_time = time_dict[f"st{st}_end{pos}"][child_key]
        conv_ind = set.union(conv_ind, set(eval(child_comb)))
        assert imp != None
        imp_sum += imp
        int_time_sum += int(flt_time * prec)
        flt_time_sum += flt_time
        st = pos
    assert imp_sum == dp_tab[(time_limit, act_num)]

    return (act_ind, conv_ind, children_dict, imp_sum, int_time_sum, flt_time_sum)


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
        model: DepthLayerResNet = ResNet34()
        model.__class__ = DepthLayerResNet
    elif args.arch == "resnet50":
        model: DepthLayerResNet = ResNet50()
        model.__class__ = DepthLayerResNet
    elif args.arch == "mobilenetv2":
        model: DepthLayerMobileNetV2 = MobileNetV2()
        model.__class__ = DepthLayerMobileNetV2
    elif args.arch == "mobilenetv2_w1.4":
        model: DepthLayerMobileNetV2 = MobileNetV2(width_mult=1.4)
        model.__class__ = DepthLayerMobileNetV2
    elif args.arch == "ddpm_cifar10":
        model: DepthLayerDDPMModel = DDPMModel(dataset2config("cifar10"))
        model.__class__ = DepthLayerDDPMModel
    elif args.arch == "ddpm_cifar10_pruned":
        assert args.pruned_path is not None
        # To handle torch.load() and load the pruned model
        import sys
        sys.path.insert(0, '../../Diff-Pruning/exp_code')
        print("Loading checkpoint {}".format(args.pruned_path))
        device = torch.device("cuda")
        states = torch.load(args.pruned_path, map_location=device)
        model = states[0].to(device)
        model: DepthLayerDDPMModel = model
        model.__class__ = DepthLayerDDPMModel
    else:
        raise NotImplementedError()

    model.morph()
    model.trace()

    blks = valid_blks(model)
    act_num = model.depth - 1
    default_ind = set(range(model.depth)) - model.iden_pos

    if args.mode == "time":
        generate_time_table(model, blks, args.dir, args.tag)
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

        act_ind, conv_ind, children, imp_sum, int_time, flt_time = optimal_patterns(
            flt_time_limit=args.time_limit,
            act_num=act_num,
            time_path=args.time_path,
            imp_path=args.imp_path,
            prec=args.prec,
            verbose=True,
        )

        act_ind = set.union({0}, act_ind)
        conv_ind = set.union({0}, conv_ind)

        nonlinear_act_ind = set.intersection(act_ind, default_ind)
        bf_ind = 0
        for ind in sorted(list(act_ind)):
            if ind > bf_ind + 1:
                nonlinear_act_ind.add(ind)
            bf_ind = ind

        result = "\n---------------------------\n"

        result += f"ACT POS       : {sorted(list(act_ind))}\n"
        result += f"NONLIN ACT POS: {sorted(list(nonlinear_act_ind))}\n"
        result += f"NUM OF NONLIN : {len(nonlinear_act_ind)}\n"
        result += f"CONV POS      : {sorted(list(conv_ind))}\n"
        result += f"CHILDREN      : \n"
        for ind, (key, comb) in children.items():
            result += f"                {ind:>3} | {key}, {comb}\n"
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
                "act_ind": act_ind,
                "conv_ind": conv_ind,
                "children_dict": children,
                "imp_sum": imp_sum,
                "int_time_sum": int_time,
                "flt_time_sum": flt_time,
            },
            os.path.join(args.checkpoint, "checkpoint.pth"),
        )
    else:
        raise NotImplementedError()
