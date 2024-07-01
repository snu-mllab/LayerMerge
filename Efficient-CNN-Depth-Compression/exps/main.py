# Code started from  https://github.com/d-li14/mobilenetv2.pytorch.git
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import pandas as pd
import copy

from models.imagenet import models, blocks
from models.model_op import reset_layers, valid_blks

from config.arguments import args, write_filename, log_filename, logs_name, log_header
from utils import Logger, train, validate
from utils.loaders import get_train_loader, get_val_loader
from utils.misc import (
    save_checkpoint,
    load_checkpoint,
    print_act_pos,
    cp_state,
    log_tool,
    KLLossSoft,
)
from tensorboardX import SummaryWriter
from itertools import product
from timm.loss import LabelSmoothingCrossEntropy

from layer_merge.models.mobilenetv2_merged_layer import make_depth_layer_mobilenet_v2
from layer_merge.models.mobilenetv2_layer import make_layer_mobilenet_v2


def random_seed(args):
    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True


def select_model(args, arch):
    print(os.path.join(args.checkpoint, args.filename))
    print("=> creating model '{}'".format(arch))
    if arch in ["learn_mobilenet_v2"]:
        model = models[arch](
            num_classes=args.nclass, width_mult=args.width_mult, add_relu=True
        )
    elif arch in ["mobilenet_v2", "dep_shrink_mobilenet_v2"]:
        model = models[arch](num_classes=args.nclass, width_mult=args.width_mult)
    elif arch in ["vgg19", "learn_vgg19"]:
        model = models[arch](num_classes=args.nclass)
    elif arch in ["depth_layer_mobilenet_v2"]:
        state = torch.load(args.act_path)
        model = make_depth_layer_mobilenet_v2(
            act_ind=state["act_ind"],
            conv_ind=state["conv_ind"],
            num_classes=args.nclass,
            width_mult=args.width_mult,
        )
    elif arch in ["layer_mobilenet_v2"]:
        state = torch.load(args.act_path)
        model = make_layer_mobilenet_v2(
            conv_ind=state["conv_ind"],
            num_classes=args.nclass,
            width_mult=args.width_mult,
        )
    else:
        raise NotImplementedError(f"Architecture {arch} not supported")

    model = torch.nn.DataParallel(model).cuda()
    return model


def load_resume(args, model, optimizer, logger):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint["epoch"]
    best_prec1 = checkpoint["best_prec1"]
    source_state = load_checkpoint(model, args.arch, args.resume, logger=logger)
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"=> loaded checkpoint '{args.resume}' (epoch {args.start_epoch})")
    args.checkpoint = os.path.dirname(args.resume)
    return best_prec1, source_state


def init_training(args, model, optimizer):
    title = "ImageNet-" + args.arch
    if args.resume:
        fname = os.path.join(os.path.dirname(args.resume), log_filename)
    else:
        fname = os.path.join(args.checkpoint, log_filename)
    logger = Logger(
        fpath=fname,
        title=title,
        resume=bool(args.resume),
    )

    if args.resume:
        logger.comment("")
        logger.comment("=================")
        logger.comment("")
        logger.comment("Resuming...")
        logger.comment("")
        best_prec1, source_state = load_resume(args, model, optimizer, logger)
    else:
        best_prec1, source_state = 0, dict()

        if args.ds_pattern != "none":
            ds_pat = (args.ds_pattern, args.compress_k)
            assert args.pretrain
        else:
            ds_pat = None
        path = args.pretrain if args.pretrain else None
        act_path = args.act_path if args.act_path else path

        if args.mode != "train":
            source_state = load_checkpoint(
                model, args.arch, path, act_path, ds_pat, logger
            )

    logger.set_names(log_header)

    log_tool(str(optimizer), logger, "opt")
    return logger, best_prec1, source_state


def set_loader(args, holdout=False):
    assert args.dataset in ["imagenet"]
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


def action(model, source_state, mode, args, val_loader, criterion):
    if mode == "eval":
        print(model)
        print_act_pos(model.module, source_state)
        val_loss, prec1 = validate(val_loader, model, criterion, args)
        print(f"Validation loss : {val_loss:.2f} | Top 1 Accuracy : {prec1}")
    elif mode == "merge":
        model.to("cpu")
        module = model.module
        save_state = {}
        for name in ["act_pos", "merge_pos", "compress_k"]:
            cp_state(save_state, source_state, name)

        if args.filename.endswith(".pth"):
            filename = args.filename[:-4]
        filename = f"{filename}_merged.pth"

        print(model)
        if "merge_pos" in save_state:
            m_pos = save_state["merge_pos"]
        else:
            m_pos = save_state["act_pos"]

        if args.arch in [
            "learn_mobilenet_v2",
            "learn_vgg19",
        ]:
            module.merge(save_state["act_pos"], m_pos)
            print(model)
            print()
            print(f"act_pos             : {save_state['act_pos']}")
            if "merge_pos" in save_state:
                print(f"merge_pos           : {m_pos}")
            print()
        elif args.arch == "dep_shrink_mobilenet_v2":
            module.merge(save_state["act_pos"])
            print(model)

        save_state.update({"state_dict": model.state_dict(), "merged": True})
        save_checkpoint(
            save_state,
            False,
            checkpoint=args.checkpoint,
            filename=filename,
        )


def prepare_finetune(model, args, source_state):
    model.to("cpu")
    module = model.module
    save_state = dict()
    cp_state(save_state, source_state, "act_pos")
    cp_state(save_state, source_state, "merge_pos")
    cp_state(save_state, source_state, "compress_k")

    # For `learn_mobilenet_v2`, see `utils.misc.load_checkpoint` ftn
    if args.arch == "dep_shrink_mobilenet_v2":
        act_pos, _ = module.get_act_info()
        module.fix_act(act_pos)
        save_state["act_pos"] = act_pos

    model.to("cuda")
    return save_state


def prepare_train(model, args):
    module = model.module
    save_state = dict()
    if args.arch == "dep_shrink_mobilenet_v2":
        module.compress_k = args.compress_k
        save_state["compress_k"] = args.compress_k
        module.set_act_hats()
    return save_state


def run_one_epoch(
    args,
    epoch,
    model,
    logger,
    writer,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    best_prec1,
    save_state,
):

    print("\nEpoch: [%d | %d]" % (epoch + 1, args.epochs))
    if args.mode == "train":
        if args.arch == "dep_shrink_mobilenet_v2":
            with torch.no_grad():
                str_act_param = str(torch.cat(model.module.get_arch_parameters()))
            log_tool(str_act_param, logger, "ap")

    # train for one epoch
    train_loss, train_acc = train(
        train_loader,
        model,
        criterion,
        optimizer,
        epoch,
        args.epochs,
        args.lr_decay,
        args.reg,
        logger,
        args,
    )

    # evaluate on validation set
    val_loss, prec1 = validate(val_loader, model, criterion, args)
    lr = optimizer.param_groups[0]["lr"]

    # append logger file
    logger.append([epoch + 1, lr, train_loss, val_loss, train_acc, prec1])

    # tensorboardX
    writer.add_scalar("learning_rate", lr, epoch + 1)
    writer.add_scalars(
        "loss", {"train loss": train_loss, "validation loss": val_loss}, epoch + 1
    )
    writer.add_scalars(
        "accuracy",
        {"train accuracy": train_acc, "validation accuracy": prec1},
        epoch + 1,
    )

    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_state.update(
        {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec1,
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": prec1,
        }
    )

    save_checkpoint(
        save_state,
        is_best,
        checkpoint=args.checkpoint,
        filename=write_filename,
    )
    return best_prec1


def train_finetune(model, criterion, optimizer):
    # set logging file config
    mode = args.mode
    logger, best_prec1, source_state = init_training(args, model, optimizer)
    if not args.ft_holdout:
        train_loader, _, val_loader = set_loader(args)
        vloader = val_loader
    else:
        train_loader, holdout_loader, test_loader = set_loader(args, holdout=True)
        vloader = holdout_loader

    # Mode dependant action of the script
    if mode == "train":
        save_state = prepare_train(model, args)
    elif mode == "finetune":
        save_state = prepare_finetune(model, args, source_state)
        if "act_pos" in save_state:
            logger.comment(str(save_state["act_pos"]))

    logger.comment(str(model))

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, logs_name))
    for epoch in range(args.start_epoch, args.epochs):
        best_prec1 = run_one_epoch(
            args,
            epoch,
            model,
            logger,
            writer,
            train_loader,
            vloader,
            criterion,
            optimizer,
            best_prec1,
            save_state,
        )
        if args.ft_holdout:
            test_loss, test_acc = validate(test_loader, model, criterion, args)
            logger.comment(f"Test Acc: {test_acc} | Test Loss : {test_loss}")
            logger.comment("-----------------------------")

    logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, "log.eps"))
    writer.close()
    print("Best accuracy:")
    print(best_prec1)


def eval_merge(model, criterion):
    # set logging file config
    mode = args.mode
    path = os.path.join(args.checkpoint, args.filename)
    if args.act_path:
        act_path = args.act_path
    else:
        act_path = path
    source_state = load_checkpoint(model, args.arch, path, act_path)
    if args.arch in ["depth_layer_mobilenet_v2", "layer_mobilenet_v2"]:
        model.module = model.module.merge()
    _, _, val_loader = set_loader(args)
    # Mode dependant action of the script
    action(model, source_state, mode, args, val_loader, criterion)


def measure_imp(model, criterion, optimizer):
    logger, _, source_state = init_training(args, model, optimizer)
    train_loader, holdout_loader, test_loader = set_loader(args, holdout=True)
    module = model.module

    act_num = module.get_act_info()[1]
    epoch = 0

    assert args.arch in ["learn_mobilenet_v2", "learn_vgg19"]
    if args.arch in ["learn_vgg19"]:
        default = set(range(1, act_num + 1))
    elif args.arch == "learn_mobilenet_v2":
        default = set(range(1, act_num + 1)) - set(range(2, act_num + 1, 3))
    model.module.fix_act(default)

    writer = SummaryWriter(os.path.join(args.checkpoint, logs_name))
    if all([st in source_state for st in ["holdout_train_loss", "holdout_train_acc"]]):
        train_loss, train_acc = (
            source_state["holdout_train_loss"],
            source_state["holdout_train_acc"],
        )
    else:
        print("Measuring base performance in train-set")
        train_loss, train_acc = validate(train_loader, model, criterion, args)
    if all(st in source_state for st in ["holdout_val_loss", "holdout_val_acc"]):
        val_loss, val_acc = (
            source_state["holdout_val_loss"],
            source_state["holdout_val_acc"],
        )
    else:
        print("Measuring base performance in holdout-val-set")
        val_loss, val_acc = validate(holdout_loader, model, criterion, args)

    logger.append([epoch, args.lr, train_loss, val_loss, train_acc, val_acc])
    logger.comment("")

    base_tl, base_ta, base_vl, base_va = train_loss, train_acc, val_loss, val_acc
    blks = valid_blks(model.module)

    blk_pos = list(model.module.get_blk_info()[0].values())
    # Extended blocks have (st, end, st_act, end_act)
    ext_blks = []
    for st, end in blks:
        # Exclude blk_pos blocks that ends with zero
        # when both st and end is blk_pos
        if args.arch == "learn_mobilenet_v2":
            if args.exclude_zeros:
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
        elif args.arch == "learn_vgg19":
            st_acts, end_acts = [1], [1]
        else:
            raise NotImplementedError()
        for st_act, end_act in product(st_acts, end_acts):
            ext_blks.append((st, end, st_act, end_act))

    logger.comment(f"Number of extended blocks are {len(ext_blks)}.")
    if args.from_blk != None and args.to_blk != None:
        logger.comment(f"Training blocks from {args.from_blk} to {args.to_blk}")
    logger.comment("")

    for ind, (st, end, st_act, end_act) in enumerate(ext_blks):
        if args.from_blk != None and args.to_blk != None:
            if ind < args.from_blk or args.to_blk <= ind:
                continue
        if os.path.exists(os.path.join(args.checkpoint, f"importance{ind}.csv")):
            logger.comment(f"Skipping block {ind:>3} since the csv file exists...")
            continue
        tmp_model = copy.deepcopy(model)
        logger.comment("")
        logger.comment(f"blk  : {(st, end, st_act, end_act)}")

        if st + 1 == end:
            if "mobilenet_v2" in args.arch:
                reset_layers(
                    tmp_model.module.features[1:-1], blocks[args.arch], st, end, logger
                )
            elif "vgg19" in args.arch:
                reset_layers(
                    tmp_model.module.features, blocks[args.arch], st, end, logger
                )

        new_acts = set()
        for i in default:
            if st < i and i < end:
                continue
            new_acts = set.union(new_acts, {i})
        if st_act:
            new_acts = set.union(new_acts, {st})
        if end_act:
            new_acts = set.union(new_acts, {end})

        logger.comment(f"acts : {new_acts}")
        tmp_model.module.fix_act(new_acts)

        tmp_optimizer = torch.optim.SGD(
            tmp_model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

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
                args.reg,
                logger,
                args,
            )
            val_loss, val_acc = validate(holdout_loader, tmp_model, criterion, args)
            lr = tmp_optimizer.param_groups[0]["lr"]
            # append logger file
            logger.append([epoch + 1, lr, train_loss, val_loss, train_acc, val_acc])

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
        csv_path = os.path.join(args.checkpoint, f"importance{ind}.csv")
        blks.to_csv(csv_path, sep=",", index=False)


def main():
    random_seed(args)
    model = select_model(args, args.arch)
    # define loss function (criterion) and optimizer
    if args.distill > 0:
        if args.arch in ["learn_mobilenet_v2", "dep_shrink_mobilenet_v2"]:
            teacher_arch = "mobilenet_v2"
        else:
            raise NotImplementedError
        teacher_model = models[teacher_arch](
            num_classes=args.nclass, width_mult=args.width_mult
        )
        teacher_model = torch.nn.DataParallel(teacher_model).cuda()
        pretrain_state = torch.load(args.pretrain)
        teacher_model.load_state_dict(pretrain_state["state_dict"])
        del pretrain_state
        criterion = KLLossSoft(alpha=args.distill, teacher=teacher_model)
    elif args.aug:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    if args.mode in ["train", "finetune"]:
        train_finetune(model, criterion, optimizer)
    elif args.mode in ["eval", "merge"]:
        eval_merge(model, criterion)
    elif args.mode in ["dp_imp"]:
        measure_imp(model, criterion, optimizer)
    else:
        raise NotImplementedError("Add mode")


if __name__ == "__main__":
    main()
