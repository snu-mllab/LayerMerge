import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Inference Time with TensorRT")
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    help="directory name",
)
parser.add_argument(
    "-n",
    "--num",
    type=int,
    help="the number of blks",
)
import re


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


if __name__ == "__main__":
    args = parser.parse_args()
    res = pd.DataFrame()
    for currentpath, folders, files in os.walk(args.dir):
        for f in sorted(files, key=natural_key):
            if ".csv" in f:
                print(f)
                tmp = pd.read_csv(os.path.join(currentpath, f))
                res = pd.concat([res, tmp])
    print(len(res))
    assert len(res) == args.num
    res.to_csv(os.path.join(args.dir, "importance.csv"))
