import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import pandas as pd
import copy
import argparse
import pkg_resources
from torch.utils.data import Subset

import layer_merge.kim23efficient.datasets as datasets

from itertools import product
from timm.data import create_transform
from layer_merge.trainer import train, validate
from layer_merge.ddpm_trainer import ddpm_train, ddpm_validate
from layer_merge.kim23efficient.generate_tables import valid_blks

from layer_merge.models.merge_op import reset_layers, reset_convs
from layer_merge.models.resnet import ResNet34, ResNet50
from layer_merge.models.resnet_merged import DepthResNet
from layer_merge.models.ddpm import DDPMModel
from layer_merge.models.ddpm_merged import DepthDDPMModel
from layer_merge.models.ddpm_datasets import get_dataset, dataset2config


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


def get_train_loader(
    data_path,
    batch_size,
    workers=5,
    _worker_init_fn=None,
    input_size=224,
    nclass=1000,
    holdout=None,
    timm_aug=False,
):
    traindir = os.path.join(data_path, "train")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if holdout == "val":
        aug = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        if timm_aug:
            aug = create_transform(
                input_size=224,
                is_training=True,
                color_jitter=0.4,
                auto_augment="rand-m5-mstd0.5-inc1",
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
                interpolation="bicubic",
            )
        else:
            aug = transforms.Compose(
                [
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    train_dataset = datasets.ImageFolder(traindir, aug, nclass=nclass, holdout=holdout)

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if holdout == "val":
        shfl = False
    else:
        shfl = train_sampler is None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=shfl,
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )

    return train_loader, len(train_loader)


def get_val_loader(
    data_path, batch_size, workers=5, _worker_init_fn=None, input_size=224, nclass=1000
):
    valdir = os.path.join(data_path, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        nclass=nclass,
    )

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )

    return val_loader, len(val_loader)


def set_loader(args, holdout=False):
    train_loader, _ = get_train_loader(
        args.data,
        args.batch_size,
        workers=args.workers,
        input_size=args.input_size,
        nclass=args.nclass,
        holdout="train" if holdout else None,
        timm_aug=args.aug,
    )
    if holdout:
        holdout_loader, _ = get_train_loader(
            args.data,
            100,
            workers=args.workers,
            input_size=args.input_size,
            nclass=args.nclass,
            holdout="val",
        )
    else:
        holdout_loader = None
    val_loader, _ = get_val_loader(
        args.data,
        100,
        workers=args.workers,
        input_size=args.input_size,
        nclass=args.nclass,
    )
    return train_loader, holdout_loader, val_loader


def set_ddpm_loader(args, holdout=False):
    assert args.arch.startswith("ddpm")
    dataset = args.arch.split("_")[1]
    config = dataset2config(dataset)
    train_dataset, test_dataset = get_dataset(args.data, config)
    holdout_dataset = None

    if holdout:
        file_path = pkg_resources.resource_filename(
            "layer_merge.models.ddpm_datasets", f"{dataset}_holdout_val.txt"
        )
        holdout_val_indices = []
        with open(file_path, 'r') as file:
            for line in file:
                holdout_val_indices.append(int(line.strip()))
        
        holdout_train_indices = []
        file_path = pkg_resources.resource_filename(
            "layer_merge.models.ddpm_datasets", f"{dataset}_holdout_train.txt"
        )
        with open(file_path, 'r') as file:
            for line in file:
                holdout_train_indices.append(int(line.strip()))

        holdout_dataset = copy.deepcopy(train_dataset)
        holdout_dataset.transform = test_dataset.transform
        holdout_dataset = Subset(holdout_dataset, holdout_val_indices)
        train_dataset = Subset(train_dataset, holdout_train_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    if holdout_dataset:
        holdout_loader = torch.utils.data.DataLoader(
            holdout_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        holdout_loader = None
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, holdout_loader, val_loader


def measure_imp(args, model, criterion):
    source_state = torch.load(args.pretrained)
    module = model.module
    module.load_state_dict(source_state["state_dict"])
    train_loader, holdout_loader, test_loader = set_loader(args, holdout=True)

    act_num = module.depth - 1
    epoch = 0

    default = set(range(act_num + 1))
    model.module.fix_act(default)

    if all(
        st in source_state
        for st in ["kim23efficient_hl_train_loss", "kim23efficient_hl_train_acc"]
    ):
        train_loss, train_acc = (
            source_state["kim23efficient_hl_train_loss"],
            source_state["kim23efficient_hl_train_acc"],
        )
    else:
        print("Measuring base performance in train-set")
        train_loss, train_acc = validate(train_loader, model, criterion, args)
    if all(
        st in source_state
        for st in ["kim23efficient_hl_val_loss", "kim23efficient_hl_val_acc"]
    ):
        val_loss, val_acc = (
            source_state["kim23efficient_hl_val_loss"],
            source_state["kim23efficient_hl_val_acc"],
        )
    else:
        print("Measuring base performance in holdout-val-set")
        val_loss, val_acc = validate(holdout_loader, model, criterion, args)

    # source_state["kim23efficient_hl_train_loss"] = train_loss
    # source_state["kim23efficient_hl_train_acc"] = train_acc
    # source_state["kim23efficient_hl_val_loss"] = val_loss
    # source_state["kim23efficient_hl_val_acc"] = val_acc
    # torch.save(source_state, args.pretrained)
    # exit()

    print(
        f"Epoch : {epoch + 1:>4}   | LR    : {args.lr:>4}   ",
        f"| Train loss : {train_loss:.6f}   | Val  loss : {val_loss:.6f}   | ",
        f"| Train acc  : {train_acc:.3f}    | Val  acc  : {val_acc:.3f}    | ",
    )
    print("")

    base_tl, base_ta, base_vl, base_va = train_loss, train_acc, val_loss, val_acc
    blks = valid_blks(model.module)

    # Extended blocks have (st, end, st_act, end_act)
    ext_blks = []
    for st, end in blks:
        st_acts, end_acts = [1], [1]
        for st_act, end_act in product(st_acts, end_acts):
            ext_blks.append((st, end, st_act, end_act))

    print(f"Number of extended blocks are {len(ext_blks)}.")
    if args.from_blk != None and args.to_blk != None:
        print(f"Training blocks from {args.from_blk} to {args.to_blk}")
    print("")

    for ind, (st, end, st_act, end_act) in enumerate(ext_blks):
        if args.from_blk != None and args.to_blk != None:
            if ind < args.from_blk or args.to_blk <= ind:
                continue
        if os.path.exists(os.path.join(args.save_path, f"importance{ind}.csv")):
            print(f"Skipping block {ind:>3} since the csv file exists...")
            continue
        tmp_model = copy.deepcopy(model)
        print("")
        print(f"blk  : {(st, end, st_act, end_act)}")

        if st + 1 == end:
            reset_layers(tmp_model.module.convs, tmp_model.module.bns, st, end)

        new_acts = set()
        for i in default:
            if st < i and i < end:
                continue
            new_acts = set.union(new_acts, {i})
        if st_act:
            new_acts = set.union(new_acts, {st})
        if end_act:
            new_acts = set.union(new_acts, {end})

        print(f"acts : {new_acts}")
        tmp_model.module.fix_act(new_acts)
        tmp_model.module.adjust_padding()

        tmp_optimizer = torch.optim.SGD(
            tmp_model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

        print(tmp_model)
        n = args.imp_epoch
        for i in range(n):
            train_loss, train_acc = train(
                train_loader,
                tmp_model,
                criterion,
                tmp_optimizer,
                i,
                n,
                args.imp_lr_decay,
                args,
            )
            val_loss, val_acc = validate(holdout_loader, tmp_model, criterion, args)
            lr = tmp_optimizer.param_groups[0]["lr"]
            # append logger file
            print(
                f"Epoch : {epoch + 1:>4}   | LR    : {lr:>4}   ",
                f"| Train loss : {train_loss:.6f}   | Val  loss : {val_loss:.6f}   | ",
                f"| Train acc  : {train_acc:.3f}    | Val  acc  : {val_acc:.3f}   | ",
            )

        dtl, dta = train_loss - base_tl, train_acc - base_ta
        dvl, dva = val_loss - base_vl, val_acc - base_va

        del tmp_model
        del tmp_optimizer

        blks_dict = {
            "id": [ind],
            "st": [st],
            "end": [end],
            "st_act": [st_act],
            "end_act": [end_act],
            "train_loss": [dtl],
            "train_acc": [dta],
            "val_loss": [dvl],
            "val_acc": [dva],
        }
        blks = pd.DataFrame(blks_dict)
        os.makedirs(args.save_path, exist_ok=True)
        csv_path = os.path.join(args.save_path, f"importance{ind}.csv")
        blks.to_csv(csv_path, sep=",", index=False)


def measure_imp_diffusion(args, model):
    if args.pretrained:
        source_state = torch.load(args.pretrained)
        module = model.module
        if all(k.startswith("module.") for k in source_state):
            model.load_state_dict(source_state)
        else:
            module.load_state_dict(source_state)
    else:
        assert args.pruned_path is not None
        module = model.module
    train_loader, holdout_loader, test_loader = set_ddpm_loader(args, holdout=True)

    act_num = module.depth
    epoch = 0

    default_ind = set(range(1, act_num + 1)) - module.iden_pos
    model.module.fix_act(default_ind)

    print("Measuring base performance in train-set")
    train_loss = ddpm_validate(train_loader, model.module)

    print("Measuring base performance in holdout-val-set")
    val_loss = ddpm_validate(holdout_loader, model.module)

    print(
        f"Epoch : {epoch + 1:>4}   | LR    : {args.lr:>4}   ",
        f"| Train loss : {train_loss:.6f}   | Val  loss : {val_loss:.6f}   | ",
    )
    print("")

    base_tl, base_vl = train_loss, val_loss
    blks = valid_blks(model.module)

    # Extended blocks have (st, end, st_act, end_act)
    ext_blks = []
    for st, end in blks:
        st_acts, end_acts = [1], [1]
        for st_act, end_act in product(st_acts, end_acts):
            ext_blks.append((st, end, st_act, end_act))

    print(f"Number of blocks are {len(ext_blks)}.")
    if args.from_blk != None and args.to_blk != None:
        print(f"Training blocks from {args.from_blk} to {args.to_blk}")
    print("")

    for ind, (st, end, st_act, end_act) in enumerate(ext_blks):
        if args.from_blk != None and args.to_blk != None:
            if ind < args.from_blk or args.to_blk <= ind:
                continue
        if os.path.exists(os.path.join(args.save_path, f"importance{ind}.csv")):
            print(f"Skipping block {ind:>3} since the csv file exists...")
            continue
        tmp_model = copy.deepcopy(model)
        print("")
        print(f"blk  : {(st, end, st_act, end_act)}")

        if st + 1 == end:
            reset_convs(tmp_model.module.convs, st, end)

        new_acts = set()
        for i in default_ind:
            if st < i and i < end:
                continue
            new_acts = set.union(new_acts, {i})
        if st_act:
            new_acts = set.union(new_acts, {st})
        if end_act:
            new_acts = set.union(new_acts, {end})


        print(f"acts : {new_acts}")
        # fix_act should follow after the fix_conv
        tmp_model.module.fix_act(new_acts)

        tmp_optimizer = torch.optim.Adam(
            tmp_model.parameters(), 
            lr=tmp_model.module.config.optim.lr, 
            weight_decay=tmp_model.module.config.optim.weight_decay,
            betas=(tmp_model.module.config.optim.beta1, 0.999), 
            amsgrad=tmp_model.module.config.optim.amsgrad,
            eps=tmp_model.module.config.optim.eps,
        )

        print(tmp_model)
        n = args.imp_epoch
        for i in range(n):
            print(f"Epoch {i+1}/{n} on train subset (n=500)")
            train_loss = ddpm_train(train_loader, tmp_model.module, tmp_optimizer)
        val_loss = ddpm_validate(holdout_loader, tmp_model.module)
        # append logger file
        print(
            f"| Train loss : {train_loss:.6f}   | Val  loss : {val_loss:.6f}   | ",
        )

        dtl = train_loss - base_tl
        dvl = val_loss - base_vl

        del tmp_model
        # del tmp_optimizer

        blks_dict = {
            "id": [ind],
            "st": [st],
            "end": [end],
            "st_act": [st_act],
            "end_act": [end_act],
            "train_loss": [dtl],
            "train_loss_ratio": [dtl / base_tl],
            "val_loss": [dvl],
            "val_loss_ratio": [dvl / base_vl],
        }
        blks = pd.DataFrame(blks_dict)
        os.makedirs(args.save_path, exist_ok=True)
        csv_path = os.path.join(args.save_path, f"importance{ind}.csv")
        blks.to_csv(csv_path, sep=",", index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Script for generating importance table"
    )
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        choices=[
            "resnet34", 
            "resnet50", 
            "ddpm_cifar10",
            "ddpm_cifar10_pruned",
        ],
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="/ssd_data/imagenet",
        help="ImageNet directory name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=1000,
        help="Batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Workers",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input resolution",
    )
    parser.add_argument(
        "--nclass",
        type=int,
        default=1000,
        choices=[10, 100, 1000],
        help="number of classes",
    )
    parser.add_argument(
        "--aug",
        type=str2bool,
        default=False,
        help="Whether to use augmentation technique",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="Path to the pre-trained weight"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Training schedule for importance"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Training schedule for importance"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Training schedule for importance",
    )
    parser.add_argument(
        "--nesterov",
        type=str2bool,
        default=True,
        help="Training schedule for importance",
    )
    parser.add_argument(
        "--imp_epoch", type=int, default=1, help="Training schedule for importance"
    )
    parser.add_argument(
        "--imp_lr_decay",
        type=str,
        default="cos",
        help="Training schedule for importance",
    )
    parser.add_argument(
        "--from_blk", type=int, default=None, help="Starting block index"
    )
    parser.add_argument("--to_blk", type=int, default=None, help="Ending block index")
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the table"
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

    model = nn.DataParallel(model).cuda()

    if args.arch.startswith("ddpm"):
        measure_imp_diffusion(args, model)
    else:
        criterion = nn.CrossEntropyLoss().cuda()
        measure_imp(args, model, criterion)
