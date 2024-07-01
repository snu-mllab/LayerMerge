import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import socket
import pandas as pd
from itertools import product
from datetime import datetime
from typing import List, Tuple

from layer_merge.models.resnet import ResNet34, ResNet50
from layer_merge.models.resnet_merged import DepthResNet
from layer_merge.models.ddpm import DDPMModel
from layer_merge.models.ddpm_merged import DepthDDPMModel
from layer_merge.models.ddpm_datasets import dataset2config
from layer_merge.measure import torch_time


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


def val_df(df, st, end, st_act=None, end_act=None, col=""):
    """
    Helper function for pandas DataFrame
    """
    mask = (df["st"] == st) & (df["end"] == end)
    if st_act != None:
        mask = mask & (df["st_act"] == st_act)
    if end_act != None:
        mask = mask & (df["end_act"] == end_act)
    filtered = df.loc[mask, :]
    if len(filtered) == 0:
        return None
    assert len(filtered) == 1
    return filtered[col].values[0]


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


def simulate_list_merge(convs: List[nn.Conv2d]):
    assert all(isinstance(x, nn.Conv2d) for x in convs)
    new_in_channels = convs[0].in_channels
    new_out_channels = convs[-1].out_channels
    new_kernel = convs[0].kernel_size[0]
    for conv in convs[1:]:
        new_kernel = (new_kernel // 2 + conv.kernel_size[0] // 2) * 2 + 1
    new_stride = convs[0].stride[0]
    for conv in convs[1:]:
        new_stride = new_stride * conv.stride[0]
    new_pad = convs[0].padding[0]
    for conv in convs[1:]:
        new_pad = new_pad + conv.padding[0]
    is_dw = all(x.groups == x.in_channels for x in convs)
    new_conv = nn.Conv2d(
        new_in_channels,
        new_out_channels,
        new_kernel,
        new_stride,
        new_pad,
        groups=convs[0].in_channels if is_dw else 1,
    )
    return new_conv


def generate_time_table(
    model: DepthResNet,
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
    blks_dict = {"id": [], "st": [], "end": [], "time": [], "stdev": []}
    for ind, (st, end) in enumerate(blks):

        now = datetime.now().strftime("%m/%d %H:%M:%S")
        print(f"ind : {ind:>3} | st : {st:>2} | end : {end:>2} | {now}")
        blks_dict["id"].append(ind)
        blks_dict["st"].append(st)
        blks_dict["end"].append(end)

        conv_lst = get_conv_lst(model.convs, st, end)

        print(conv_lst)
        merged_conv = simulate_list_merge(conv_lst)
        print(merged_conv)
        merged_model = nn.Sequential(merged_conv, nn.ReLU(inplace=True))

        time, std = torch_time(
            merged_model,
            out_shape[st],
            f"st : {st:>2} | end : {end:>2}",
            True,
            rep=200,
            warmup=300,
        )
        blks_dict["time"].append(time)
        blks_dict["stdev"].append(std)

    blks = pd.DataFrame(blks_dict)
    if tag:
        filename = f"time_{tag}.csv"
    else:
        filename = f"time_{env}.csv"
    csv_path = os.path.join(dir, filename)
    blks.to_csv(csv_path, sep=",", index=False)


def generate_optimal_time_table(
    blks: List[Tuple[int, int]],
    dir: str,
    tag: str,
    time_path="",
):
    """
    This function computes the $T_{opt}(\cdot, \cdot)$ table in the paper
    """
    df = pd.read_csv(time_path)
    # In order to speed up the `isin` operation
    blks_set = set(blks.copy())

    df_time_dict = dict()
    df_time_stdev_dict = dict()
    for st, end in blks:
        df_time_dict[f"{st}_{end}"] = val_df(df, st=st, end=end, col="time")
        df_time_stdev_dict[f"{st}_{end}"] = val_df(df, st=st, end=end, col="stdev")

    opt_tab = dict()
    argmin_lst = dict()
    stdev_tab = dict()

    for st, end in blks:
        g = df_time_dict[f"{st}_{end}"]
        stdev = df_time_stdev_dict[f"{st}_{end}"]
        assert g != None
        min_val = g
        amin = end
        for j in range(st + 1, end):
            if (st, j) in blks_set and (j, end) in blks_set:
                g_j = df_time_dict[f"{j}_{end}"]
                std_j = df_time_stdev_dict[f"{j}_{end}"]
                assert g_j != None
                if opt_tab[(st, j)] + g_j < min_val:
                    min_val = opt_tab[(st, j)] + g_j
                    stdev = stdev_tab[(st, j)] + std_j
                    amin = j
        opt_tab[(st, end)] = min_val
        stdev_tab[(st, end)] = stdev
        if amin != end:
            argmin_lst[(st, end)] = argmin_lst[(st, amin)] + [end]
        else:
            argmin_lst[(st, end)] = [st, end]

    blks_dict = {"id": [], "st": [], "end": [], "time": [], "stdev": [], "breaks": []}
    for ind, (st, end) in enumerate(blks):
        blks_dict["id"].append(ind)
        blks_dict["st"].append(st)
        blks_dict["end"].append(end)
        blks_dict["time"].append(opt_tab[(st, end)])
        blks_dict["stdev"].append(stdev_tab[(st, end)])
        blks_dict["breaks"].append(",".join([str(i) for i in argmin_lst[(st, end)]]))

    blks = pd.DataFrame(blks_dict)
    if tag:
        filename = f"opt_time_{tag}.csv"
    else:
        filename = f"opt_time.csv"
    csv_path = os.path.join(dir, filename)
    blks.to_csv(csv_path, sep=",", index=False)


def generate_ext_imp_table(
    ext_blks: List[Tuple[int, int, int, int]],
    act_num: int,
    dir: str,
    imp_path="",
    score: str = "val_acc",
    default: set = {},
    norm: str = "default",
    alpha: float = 1.0,
):
    """
    This function computes the $I_{ext}(\cdot, \cdot, \cdot, \cdot)$ table in the paper
    """
    df_imp = pd.read_csv(imp_path)

    if norm == "default":
        # Normalize with default
        mask1 = df_imp["end"] - df_imp["st"] == 1
        mask2 = (df_imp["st_act"] == 1) == (df_imp["st"].isin(default))
        mask3 = (df_imp["end_act"] == 1) == (df_imp["end"].isin(default))
        mask4 = (df_imp["st"] == 0) & (df_imp["end"] == 1)
        mask = (mask1 & mask2 & mask3) | mask4
        assert sum(mask) == act_num
    elif norm == "single":
        mask = df_imp["end"] - df_imp["st"] == 1
    elif norm == "all":
        mask = df_imp["st_act"].isin([0, 1])
    else:
        raise NotImplementedError()

    if score == "val_loss_ratio":
        df_imp[score] = - df_imp[score]

    mean = df_imp.loc[mask, :].mean()[score]
    df_imp[score] -= alpha * mean
    print(mean)
    print(alpha * mean)

    df_imp_dict = dict()
    for m in range(1, act_num + 1):
        for i in range(0, m):
            for j in range(2):
                for a in range(2):
                    df_imp_dict[f"{i}_{m}_{j}_{a}"] = val_df(
                        df_imp, st=i, end=m, st_act=j, end_act=a, col=score
                    )

    # In order to speed up the `isin` operation
    ext_blks_set = set(ext_blks.copy())
    ext_tab = dict()
    argmax_lst = dict()

    """
    I_{ext}(st, end, a) \leftarrow \max_j(I_{ext}(st, j, 0) + h(j, end, 0, a))
    """
    for (st, end, st_act, end_act) in ext_blks:
        h = df_imp_dict[f"{st}_{end}_{st_act}_{end_act}"]
        assert h != None
        max_val = h
        amax = end
        for j in range(st + 1, end):
            if (st, j, st_act, 0) in ext_tab and (j, end, 0, end_act) in ext_blks_set:
                h_j = df_imp_dict[f"{j}_{end}_{0}_{end_act}"]
                assert h_j != None
                if ext_tab[(st, j, st_act, 0)] + h_j > max_val:
                    max_val = ext_tab[(st, j, st_act, 0)] + h_j
                    amax = j
        ext_tab[(st, end, st_act, end_act)] = max_val
        if amax != end:
            sub = argmax_lst[(st, amax, st_act, 0)]
            argmax_lst[(st, end, st_act, end_act)] = sub + [end]
        else:
            argmax_lst[(st, end, st_act, end_act)] = [st, end]

    blks_dict = {
        "id": [],
        "st": [],
        "end": [],
        "st_act": [],
        "end_act": [],
        "imp": [],
        "breaks": [],
    }
    for ind, (st, end, st_act, end_act) in enumerate(ext_blks):
        blks_dict["id"].append(ind)
        blks_dict["st"].append(st)
        blks_dict["st_act"].append(st_act)
        blks_dict["end"].append(end)
        blks_dict["end_act"].append(end_act)
        blks_dict["imp"].append(ext_tab[(st, end, st_act, end_act)])
        blks_dict["breaks"].append(
            ",".join([str(i) for i in argmax_lst[(st, end, st_act, end_act)]])
        )

    blks = pd.DataFrame(blks_dict)

    filename = f"ext_importance"
    filename += f"_s_{score}"
    if norm != "default":
        filename += f"_n_{norm}"
    if alpha != 1.0:
        filename += f"_a_{alpha}"
    filename += ".csv"
    csv_path = os.path.join(dir, filename)
    blks.to_csv(csv_path, sep=",", index=False)


def construct_ext_blks(blks):
    ext_blks = []
    for st, end in blks:
        st_acts, end_acts = [1], [1]
        for st_act, end_act in product(st_acts, end_acts):
            ext_blks.append((st, end, st_act, end_act))
    return ext_blks


def optimal_patterns(
    flt_time_limit: float,
    act_num: int,
    opt_time_path: str,
    ext_imp_path: str,
    prec: float = 100,
    verbose: bool = False,
    logger=None,
):
    df_opt_time = pd.read_csv(opt_time_path)
    df_ext_imp = pd.read_csv(ext_imp_path)

    df_opt_time_dict = dict()
    df_opt_time_brks_dict = dict()
    df_ext_imp_dict = dict()
    for m in range(1, act_num + 1):
        for i in range(0, m):
            df_opt_time_dict[f"{i}_{m}"] = val_df(df_opt_time, st=i, end=m, col="time")
            df_opt_time_brks_dict[f"{i}_{m}"] = val_df(
                df_opt_time, st=i, end=m, col="breaks"
            )
            for j in range(2):
                for a in range(2):
                    df_ext_imp_dict[f"{i}_{m}_{j}_{a}"] = val_df(
                        df_ext_imp, st=i, end=m, st_act=j, end_act=a, col="imp"
                    )

    time_limit = int(flt_time_limit * prec)

    # Sum of maximum importance
    dp_tab = dict()
    # Sum of optimal time
    dp_time_tab = dict()
    # Initialization
    for t in range(0, time_limit):
        dp_tab[(t, 0, 1)] = 0
        dp_time_tab[(t, 0, 1)] = 0
    max_tab = dict()
    mpos_tab = dict()
    for m in range(1, act_num + 1):
        t_min = float("Inf")
        for i in range(0, m):
            g_flt = df_opt_time_dict[f"{i}_{m}"]
            if g_flt == None:
                continue
            cand = int(g_flt * prec)
            if t_min > cand:
                t_min = cand

        # Impossible time limit
        if t_min > time_limit:
            if logger:
                logger.comment(f"Impossible time limit : {time_limit}", verb=True)
            else:
                print(f"Impossible time limit : {time_limit}")
            exit()

        for t, a in product(range(t_min, time_limit + 1), range(2)):
            if verbose and (t - t_min) % 1000 == 0:
                now = datetime.now().strftime("%m/%d %H:%M:%S")
                prg = f"m = {m:>2} & a = {a:>2} : {t - t_min:>5} / {time_limit - t_min:>5}       ||      {now}"
                if logger:
                    logger.comment(prg, verb=True)
                else:
                    print(prg)

            max_imp = float("-Inf")
            for i, j in product(range(m), range(2)):
                g_flt = df_opt_time_dict[f"{i}_{m}"]
                h = df_ext_imp_dict[f"{i}_{m}_{j}_{a}"]
                if g_flt == None or h == None:
                    assert h == None
                    continue
                g = int(g_flt * prec)
                # Skip for impossible combination
                if not (t - g, i, j) in dp_tab:
                    continue
                cand = dp_tab[(t - g, i, j)] + h
                cand_time = dp_time_tab[(t - g, i, j)] + g

                # For numerical stability of importance value
                # (we round the values before comparison between float)
                r_imp, r_cand = round(max_imp, 7), round(cand, 7)
                # Choose maximum imp (on the tie we choose the one with faster time)
                if r_imp < r_cand or (r_imp == r_cand and cand_time < opt_time):
                    max_imp = cand
                    opt_time = cand_time
                    argmax_imp = (t - g, i, j)
                    brkstr = df_opt_time_brks_dict[f"{i}_{m}"]
                    brks = set([int(pos) for pos in brkstr.split(",")])
                    if i == 0:
                        mpos = brks
                    else:
                        mpos = set.union(brks, mpos_tab[(t - g, i, j)])

            # Skip for impossible combination (or combination that we do not take into account)
            if max_imp != float("-Inf"):
                dp_tab[(t, m, a)] = max_imp
                dp_time_tab[(t, m, a)] = opt_time
                max_tab[(t, m, a)] = argmax_imp
                mpos_tab[(t, m, a)] = mpos

    t, m = time_limit, act_num
    if all(not (t, m, a) in dp_tab for a in range(0, 2)):
        if logger:
            logger.comment(f"Impossible time limit : {t}", verb=True)
        else:
            print(f"Impossible time limit : {t}")
        exit()
    ends = []
    for a in range(2):
        if (t, m, a) in dp_tab:
            ends.append(dp_tab[(t, m, a)])
        else:
            ends.append(float("-Inf"))
    sol_a = np.argmax(ends)

    a = sol_a
    opt_m_pos = {m}
    blk_ends = {m}
    act_pos = {m} if bool(sol_a) else set()
    while m > 0:
        opt_m_pos = set.union(opt_m_pos, mpos_tab[(t, m, a)])
        blk_ends = set.union(blk_ends, {m})
        act_pos = set.union(act_pos, {m}) if bool(a) else act_pos
        t, m, a = max_tab[(t, m, a)]
    assert dp_tab[(t, m, a)] == 0
    opt_m_pos -= {0}

    st, st_act = 0, 1
    imp_sum, int_time_sum, flt_time_sum = 0, 0, 0
    for pos in sorted(list(blk_ends)):
        a = int(pos in act_pos)
        if pos == 0:
            continue
        imp = df_ext_imp_dict[f"{st}_{pos}_{st_act}_{a}"]
        flt_time = df_opt_time_dict[f"{st}_{pos}"]
        assert imp != None
        imp_sum += imp
        int_time_sum += int(flt_time * prec)
        flt_time_sum += flt_time
        st, st_act = pos, a
    assert imp_sum == dp_tab[(time_limit, act_num, sol_a)]

    return (act_pos, opt_m_pos, imp_sum, int_time_sum, flt_time_sum)


def optimal_merge_pattern(act_pos, act_num, time_path):
    df = pd.read_csv(time_path)

    dp_tab = {0: 0}
    min_tab = dict()
    for m in range(1, act_num + 1):
        starts = set.union(act_pos, {0})
        t_max = max([x for x in starts if x < m])
        min_time = float("Inf")
        for t in range(t_max, m):
            g = val_df(df, st=t, end=m, col="time")
            if g == None:
                continue
            cand = dp_tab[t] + g
            if min_time > cand:
                min_time = cand
                argmin_time = t
        dp_tab[m] = min_time
        min_tab[m] = argmin_time

    m = act_num
    m_pos = {m}
    while m > 0:
        m_pos = set.union(m_pos, {m})
        m = min_tab[m]

    st, stdev = 0, 0
    for pos in sorted(list(m_pos)):
        if pos == 0:
            continue
        std = val_df(df, st=st, end=pos, col="stdev")
        assert std != None
        stdev += std
        st = pos

    assert act_pos.issubset(m_pos)
    return (m_pos, dp_tab[act_num], stdev)



def main():
    parser = argparse.ArgumentParser(description="Script for dynamic programming algorithm")
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        choices=["resnet34", "resnet50", "ddpm_cifar10", "ddpm_cifar10_pruned"],
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
    parser.add_argument(
        "--mode", type=str, choices=["time", "opt-time", "ext-imp", "solve"]
    )
    parser.add_argument(
        "--time_path", type=str, default="", help="Path to the normal time table"
    )
    parser.add_argument(
        "--chk_time_path", type=str, default="", help="Path to the normal time table"
    )
    parser.add_argument(
        "--imp_path", type=str, default="", help="Path to the normal importance table"
    )
    parser.add_argument(
        "--score", type=str, default="", help="Score for the importance"
    )
    parser.add_argument(
        "--norm", type=str, default="single", help="Normalization method of score"
    )
    parser.add_argument("--alph", type=float, default=1.0, help="Alpha in normalizing")
    parser.add_argument("--prec", type=float, default=10, help="Precision in solving")
    parser.add_argument(
        "--time_limit", type=float, default=None, help="Time limit in solving"
    )
    parser.add_argument(
        "--pruned_path", type=str, default=None, help="Path to the pruned model"
    )

    args = parser.parse_args()

    if args.arch == "resnet34":
        model: DepthResNet = ResNet34()
        model.__class__ = DepthResNet
    elif args.arch == "resnet50":
        model: DepthResNet = ResNet50()
        model.__class__ = DepthResNet
    elif args.arch == "ddpm_cifar10":
        model: DepthDDPMModel = DDPMModel(dataset2config("cifar10"))
        model.__class__ = DepthDDPMModel
    elif args.arch == "ddpm_cifar10_pruned":
        assert args.pruned_path is not None
        # To handle torch.load() and load the pruned model
        import sys
        sys.path.insert(0, '../../Diff-Pruning/exp_code')
        print("Loading checkpoint {}".format(args.pruned_path))
        device = torch.device("cuda")
        states = torch.load(args.pruned_path, map_location=device)
        model = states[0].to(device)
        model: DepthDDPMModel = model
        model.__class__ = DepthDDPMModel
    else:
        raise NotImplementedError()

    model.morph()
    model.trace()

    blks = valid_blks(model)
    act_num = model.depth - 1
    default = set(range(act_num + 1))

    if args.mode == "time":
        generate_time_table(model, blks, args.dir, args.tag)
    elif args.mode == "opt-time":
        assert args.time_path
        generate_optimal_time_table(blks, args.dir, args.tag, args.time_path)
    elif args.mode == "ext-imp":
        ext_blks = construct_ext_blks(blks)
        assert args.imp_path and args.score in ["train_acc", "val_acc", "val_loss_ratio"]
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
    elif args.mode == "solve":
        args.checkpoint = os.path.join(args.dir, f"p{args.prec}_tl{args.time_limit}")

        config = "Solving DP\n"
        config += f"IMP PATH      : {args.imp_path}\n"
        config += f"TIME PATH     : {args.time_path}\n"
        config += f"TIME LIMIT    : {args.time_limit}\n"
        config += f"PRECISION     : {args.prec}\n"

        config += f"\n---------------------------\n\n"
        print(config)

        act_pos, opt_m_pos, imp_sum, int_time_sum, flt_time_sum = optimal_patterns(
            flt_time_limit=args.time_limit,
            act_num=act_num,
            opt_time_path=args.time_path,
            ext_imp_path=args.imp_path,
            prec=args.prec,
            verbose=True,
            logger=None,
        )

        if args.chk_time_path:
            m_pos, time, _ = optimal_merge_pattern(
                act_pos=act_pos, act_num=act_num, time_path=args.chk_time_path
            )
            assert opt_m_pos == m_pos
            msg = "\nOPTIMAL MERGE POS IS CORRECT\n"
            print(msg)

        act_pos = set.union({0}, act_pos)
        opt_m_pos = set.union({0}, opt_m_pos)

        result = "\n---------------------------\n"

        result += f"ACT POS       : {act_pos}\n"
        result += f"MERGE POS     : {opt_m_pos}\n"
        result += f"IMP SUM       : {imp_sum}\n"
        result += f"INT TIME SUM  : {int_time_sum}\n"
        result += f"FLT TIME SUM  : {flt_time_sum}\n"

        print(result)

        os.makedirs(args.checkpoint, exist_ok=True)

        with open(os.path.join(args.checkpoint, "result.log"), "w") as f:
            f.write(config + msg + result)
            f.write("\n")

        torch.save(
            {
                "act_pos": act_pos,
                "merge_pos": opt_m_pos,
                "imp_sum": imp_sum,
                "int_time_sum": int_time_sum,
                "flt_time_sum": flt_time_sum,
            },
            os.path.join(args.checkpoint, "checkpoint.pth"),
        )
    else:
        raise NotImplementedError()
