import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import argparse

from utils.logger import Logger
from utils.dp import optimal_patterns, optimal_merge_pattern
from utils.misc import save_checkpoint


def make_log_file(checkpoint, filename, **kwargs):
    if not os.path.isdir(checkpoint):
        os.makedirs(checkpoint, exist_ok=True)
    write_filename = filename
    log_filename = "log.txt"
    return write_filename, log_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for solving DP")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoints",
        type=str,
        metavar="PATH",
        help="path to save checkpoint (default: checkpoints)",
    )
    parser.add_argument(
        "-f",
        "--filename",
        default="checkpoint.pth",
        type=str,
        metavar="FILE",
        help="filename of the checkopint (default: checkpoint.pth)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        help="Time limit to optimize on",
    )
    parser.add_argument(
        "--act-num",
        type=int,
        help="Total number of the full activations",
    )
    parser.add_argument(
        "--time-path", type=str, help="Path to the inference time table of each block"
    )
    parser.add_argument(
        "--imp-path", type=str, help="Path to the importance table of each block"
    )
    parser.add_argument(
        "--prec",
        type=float,
        help="Precision when converting float constriant into int constraint",
    )
    parser.add_argument(
        "--chk-time-path",
        default="",
        type=str,
        help="Path to the inference time table of each block",
    )

    args = parser.parse_args()

    args.checkpoint = os.path.join(args.checkpoint, f"p{args.prec}_tl{args.time_limit}")

    title = "Solving DP"
    write_filename, log_filename = make_log_file(args.checkpoint, args.filename)
    logger = Logger(os.path.join(args.checkpoint, log_filename), title=title)

    logger.comment(f"IMP PATH      : {args.imp_path}")
    logger.comment(f"TIME PATH     : {args.time_path}")
    logger.comment(f"TIME LIMIT    : {args.time_limit}")
    logger.comment(f"PRECISION     : {args.prec}")

    logger.comment("\n---------------------------\n")

    act_pos, opt_m_pos, imp_sum, int_time_sum, flt_time_sum = optimal_patterns(
        flt_time_limit=args.time_limit,
        act_num=args.act_num,
        opt_time_path=args.time_path,
        ext_imp_path=args.imp_path,
        prec=args.prec,
        verbose=True,
        logger=logger,
    )

    if args.chk_time_path:
        m_pos, time, _ = optimal_merge_pattern(
            act_pos=act_pos, act_num=args.act_num, time_path=args.chk_time_path
        )
        assert opt_m_pos == m_pos
        logger.comment("")
        logger.comment("OPTIMAL MERGE POS IS CORRECT")

    logger.comment("\n---------------------------\n")

    logger.comment(f"ACT POS       : {act_pos}")
    logger.comment(f"MERGE POS     : {opt_m_pos}")
    logger.comment(f"IMP SUM       : {imp_sum}")
    logger.comment(f"INT TIME SUM  : {int_time_sum}")
    logger.comment(f"FLT TIME SUM  : {flt_time_sum}")

    save_checkpoint(
        {
            "act_pos": act_pos,
            "merge_pos": opt_m_pos,
            "imp_sum": imp_sum,
            "int_time_sum": int_time_sum,
            "flt_time_sum": flt_time_sum,
        },
        False,
        checkpoint=args.checkpoint,
        filename=args.filename,
    )
