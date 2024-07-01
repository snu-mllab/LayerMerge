import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import socket
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from datetime import datetime
from typing import List, Tuple
from itertools import product
from models.model_op import get_conv_lst, simulate_list_merge, valid_blks
from utils.measure import compile_and_time, torch_time
from utils.logger import Logger


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


def generate_time_table(
    model,
    blks: List[Tuple[int, int]],
    arch: str,
    blk_type,
    dir: str,
    tag: str = "",
    logger: Logger = None,
    trt: bool = True,
):
    """
    This function computes the $T(\cdot, \cdot)$ table in the paper
    """
    cnt = torch.cuda.device_count()
    env = f"{socket.gethostname()}_gpu{cnt}"
    out_shape = model.out_shape

    blks = valid_blks(model)
    blks_dict = {"id": [], "st": [], "end": [], "time": [], "stdev": []}
    for ind, (st, end) in enumerate(blks):
        if logger:
            now = datetime.now().strftime("%m/%d %H:%M:%S")
            logger.comment(
                f"ind : {ind:>3} | st : {st:>2} | end : {end:>2}    |     {now}",
                verb=True,
            )
        else:
            print(f"ind : {ind:>3} | st : {st:>2} | end : {end:>2}")
        blks_dict["id"].append(ind)
        blks_dict["st"].append(st)
        blks_dict["end"].append(end)

        if arch == "learn_mobilenet_v2":
            conv_lst = get_conv_lst(model.features[1:-1], blk_type, st, end)
        elif arch == "learn_vgg19":
            conv_lst = get_conv_lst(model.features, blk_type, st, end)
        else:
            raise NotImplementedError("Not supported architecture")

        if logger:
            logger.comment(str(conv_lst), verb=True)
        else:
            print(conv_lst)
        merged_conv = simulate_list_merge(conv_lst)
        if logger:
            logger.comment(str(merged_conv), verb=True)
        else:
            print(merged_conv)
        merged_model = nn.Sequential(merged_conv, nn.ReLU(inplace=True))
        if trt:
            time, std = compile_and_time(
                merged_model, out_shape[st], f"st : {st:>2} | end : {end:>2}"
            )
        else:
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


def optimal_patterns(
    flt_time_limit: float,
    act_num: int,
    opt_time_path: str,
    ext_imp_path: str,
    prec: float = 100,
    verbose: bool = False,
    logger: Logger = None,
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
