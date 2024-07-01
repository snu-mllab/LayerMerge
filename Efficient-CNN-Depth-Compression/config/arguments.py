import os
import argparse


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


def make_log_file(mode, checkpoint, filename, **kwargs):
    if mode in ["train", "finetune", "dp_imp"]:
        if not os.path.isdir(checkpoint):
            os.makedirs(checkpoint, exist_ok=True)
    if mode == "train":
        write_filename = filename
        log_filename = "log.txt"
        logs_name = "logs"
    elif mode == "finetune":
        if args.filename.endswith(".pth"):
            write_filename = filename[:-4]
        write_filename = write_filename + f"_ft_lr{kwargs['lr']}.pth"
        log_filename = f"log_{write_filename[:-4]}.txt"
        logs_name = f"logs_{write_filename[:-4]}"
    elif mode == "dp_imp":
        write_filename = ""
        log_filename = f"log_f{args.from_blk}_t{args.to_blk}.txt"
        logs_name = f"log_f{args.from_blk}_t{args.to_blk}"
    else:
        write_filename = ""
        log_filename = ""
        logs_name = ""
    log_header = ["Epoch", "LR", "Train Loss", "Valid Loss", "Train Acc.", "Valid Acc."]
    return write_filename, log_filename, logs_name, log_header


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument(
    "--aug", default=False, type=str2bool, help="Add augmentation to training script"
)
parser.add_argument("--distill", default=0.0, type=float, help="distillation ratio")
# data and models
parser.add_argument(
    "-d", "--data", metavar="DIR", default="/ssd_data/imagenet", help="path to dataset"
)
parser.add_argument(
    "--nclass",
    default=1000,
    type=int,
    choices=[10, 100, 1000],
    help="number of class to use",
)
parser.add_argument(
    "--dataset",
    default="imagenet",
    type=str,
    choices=["imagenet"],
    help="Type of dataset to use",
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="mobilenet_v2",
    help="model architecture: ",
)
parser.add_argument(
    "-j",
    "--workers",
    default=10,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 10)",
)
parser.add_argument(
    "--epochs", default=150, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
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
    "--width-mult", type=float, default=1.0, help="MobileNet model width multiplier."
)
parser.add_argument(
    "--input-size", type=int, default=224, help="MobileNet model input resolution"
)

# Optimizer
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-5,
    type=float,
    metavar="W",
    help="weight decay (default: 4e-5)",
    dest="weight_decay",
)
parser.add_argument("--nesterov", default=True, type=str2bool, help="nesterov for sgd")
parser.add_argument(
    "--lr-decay", type=str, default="cos", help="mode for learning rate decay"
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[45, 90, 135, 157],
    help="decrease learning rate at these epochs.",
)
parser.add_argument(
    "--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule."
)

# Options
parser.add_argument(
    "-p",
    "--print-freq",
    default=1000,
    type=int,
    metavar="N",
    help="print frequency (default: 1000)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-m",
    "--mode",
    default="train",
    type=str,
    choices=["train", "finetune", "eval", "merge", "dp_imp"],
    help="script mode : choose among (train/finetune/eval/merge/dp_imp)",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)

parser.add_argument("--debug", action="store_true", help="only heavy ones")

# Hyperparams of main experiment/baseline
parser.add_argument(
    "--pretrain", default="", type=str, help="path to the pretrained vanilla network"
)
parser.add_argument("--act-path", default="", type=str, help="path to the act pos")
parser.add_argument(
    "--ft-holdout",
    default=False,
    type=str2bool,
    help="Whether to finetune with holdout validation set",
)
# Merge & Finetune
parser.add_argument(
    "--time-path",
    default="",
    type=str,
    help="""
    Path to the time table. 
    Needed in finding optimal merging pattern using DP. 
    Will use act_pos for merging pattern if not specififed.
    """,
)
# DP
parser.add_argument(
    "--imp-epoch",
    default=1,
    type=int,
    help="""
    epoch per measuring the importance
    """,
)
parser.add_argument(
    "--imp-lr-decay",
    default="cos",
    type=str,
    help="""
    lr_decay when measuring the importance
    """,
)
parser.add_argument(
    "--from-blk",
    default=None,
    type=int,
    help="""
    ID of the block to measure from
    """,
)
parser.add_argument(
    "--to-blk",
    default=None,
    type=int,
    help="""
    ID of the block to measure until
    """,
)
parser.add_argument(
    "--exclude-zeros",
    default=False,
    type=str2bool,
    help="""
    Exclude blocks starting with 0 activation
    when both of the start and end indices are 
    end of the ResidualBlock
    """,
)
# DepthShrinker
parser.add_argument(
    "--lamb",
    default=1e-4,
    type=float,
    help="lambda value that weighs sparsity regularizer",
)
parser.add_argument(
    "--reg",
    default="none",
    choices=["none", "soft", "w1.0", "w1.4"],
    type=str,
    help="Type of regularizer to use",
)
parser.add_argument(
    "--compress-k",
    default=8,
    type=int,
    help="The number of InvertedResidualBlock to compress (min 0, max 17)",
)
# fmt: off
parser.add_argument(
    "--ds-pattern",
    default="none",
    choices=["none", "A", "B", "C", "D", "E", "F", "A10", "B10", "C10", "D10", "AR", "BR", "CR", "AR10", "BR10", "CR10", "AR_AUG", "BR_AUG", "CR_AUG", "AR10_AUG", "BR10_AUG", "CR10_AUG"],
    type=str,
    help="pre-defined pattern from the paper",
)
# fmt: on
args = parser.parse_args()

if args.dataset == "cifar":
    assert args.nclass == 10
elif args.dataset == "imagenet":
    assert args.nclass in [100, 1000]

# Some post-processing on args for each mode
if args.mode in ["train", "dp_imp"]:
    if args.arch in ["mobilenet_v2", "learn_mobilenet_v2", "dep_shrink_mobilenet_v2"]:
        args.checkpoint += f"_w{args.width_mult}"

    # Architecture specific config
    if args.mode in ["dp_imp"]:
        args.checkpoint += f"_ie{args.imp_epoch}"
        args.checkpoint += f"_ild_{args.imp_lr_decay}"
        if args.exclude_zeros:
            args.checkpoint += f"_ex"

        args.checkpoint = os.path.join(args.checkpoint, f"par")

    elif args.mode == "train":
        args.checkpoint += f"_e{args.epochs}"
        if args.arch == "dep_shrink_mobilenet_v2":
            args.checkpoint += f"_cmp{args.compress_k}"

if args.aug:
    args.checkpoint += "_aug"
if args.distill > 0:
    args.checkpoint += f"_dt{args.distill}"

if args.seed:
    args.checkpoint += f"_s{args.seed}"

if args.mode != "train" or args.arch == "mobilenet_v2" or args.reg == "none":
    args.reg = None

if args.resume:
    assert args.mode in ["train", "finetune"]
    if not os.path.isfile(args.resume):
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(args.resume))
    args.checkpoint = os.path.dirname(args.resume)

if args.ft_holdout:
    assert args.mode == "finetune"

if args.time_path:
    assert args.mode in ["finetune", "merge"]


write_filename, log_filename, logs_name, log_header = make_log_file(
    args.mode, args.checkpoint, args.filename, lr=args.lr
)

log_header = ["Epoch", "LR", "Train Loss", "Valid Loss", "Train Acc.", "Valid Acc."]
