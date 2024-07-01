# Code started from  https://github.com/d-li14/mobilenetv2.pytorch.git
import os
import time
from math import cos, pi

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.logger import AverageMeter
from utils.misc import KLLossSoft

# progress bar
# https://github.com/verigak/progress
from progress.bar import Bar as Bar

__all__ = ["accuracy", "validate", "train"]


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    total_epochs,
    lr_decay,
    reg=None,
    logger=None,
    args=None,
):
    bar = Bar("Processing", max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    logger.comment("-----------------------------")
    for ind, (input, target) in enumerate(train_loader):
        adjust_learning_rate(
            optimizer,
            epoch,
            ind,
            len(train_loader),
            total_epochs,
            args.lr,
            lr_decay,
            args.schedule,
            args.gamma,
        )

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        if isinstance(criterion, KLLossSoft):
            with torch.no_grad():
                soft_logits = criterion.teacher(input)
            loss = criterion(output, soft_logits, target)
        else:
            loss = criterion(output, target)
        if reg != None:
            loss += args.lamb * model.module.regularizer(reg)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.aug:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        plot_progress = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
            batch=ind + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.suffix = plot_progress
        bar.next()
        # For logging, print newline
        if (ind + 1) % args.print_freq == 0:
            logger.comment(plot_progress)
        # if i == len(train_loader.dataloader) - 1:
        #     top1, _ = accuracy(output, target, topk=(1, 5))
        if args.debug and ind > 5:
            break

    bar.finish()
    logger.comment("-----------------------------")
    return (losses.avg, top1.avg)


def validate(val_loader, model, criterion, args):
    bar = Bar("Processing", max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            # compute output
            output = model(input)
            if isinstance(criterion, KLLossSoft):
                soft_logits = criterion.teacher(input)
                loss = criterion(output, soft_logits, target)
            else:
                loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
            batch=i + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
        if args.debug and i > 5:
            break
    bar.finish()
    return (losses.avg, top1.avg)


def adjust_learning_rate(
    optimizer, epoch, iteration, num_iter, epochs, lr_init, lr_decay, schedule, gamma
):
    lr = optimizer.param_groups[0]["lr"]

    # # This warmup parameter is obsolete
    # wu_epoch = 5 if args.warmup else 0
    # wu_iter = wu_epoch * num_iter
    wu_epoch, wu_iter = 0, 0
    cur_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter

    if lr_decay == "step":
        lr = lr_init * (gamma ** ((cur_iter - wu_iter) / (max_iter - wu_iter)))
    elif lr_decay == "cos":
        lr = lr_init * (1 + cos(pi * (cur_iter - wu_iter) / (max_iter - wu_iter))) / 2
    elif lr_decay == "linear":
        lr = lr_init * (1 - (cur_iter - wu_iter) / (max_iter - wu_iter))
    elif lr_decay == "schedule":
        count = sum([1 for s in schedule if s <= epoch])
        lr = lr_init * pow(gamma, count)
    elif lr_decay == "const":
        lr = lr_init
    else:
        raise ValueError("Unknown lr mode {}".format(lr_decay))

    if epoch < wu_epoch:
        lr = lr_init * cur_iter / wu_iter

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
