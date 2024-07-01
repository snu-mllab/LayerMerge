# Code started from  https://github.com/d-li14/mobilenetv2.pytorch.git
import os
import time
from math import cos, pi

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


# progress bar
# https://github.com/verigak/progress
from progress.bar import Bar as Bar

__all__ = ["accuracy", "validate", "train"]


# Implementation adapted from AlphaNet - https://github.com/facebookresearch/AlphaNet
class KLLossSoft(torch.nn.modules.loss._Loss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    T: temperature
    """

    def __init__(
        self, alpha, teacher, size_average=None, reduce=None, reduction: str = "mean"
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.alpha = alpha
        self.teacher = teacher

    def forward(self, output, soft_logits, target, temperature=1.0):
        output, soft_logits = output / temperature, soft_logits / temperature
        soft_target_prob = torch.nn.functional.softmax(soft_logits, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        kd_loss = -torch.sum(soft_target_prob * output_log_prob, dim=1)
        if target is not None:
            target = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
            target = target.unsqueeze(1)
            output_log_prob = output_log_prob.unsqueeze(2)
            ce_loss = -torch.bmm(target, output_log_prob).squeeze()
            loss = (
                self.alpha * temperature * temperature * kd_loss
                + (1.0 - self.alpha) * ce_loss
            )
        else:
            loss = kd_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    total_epochs,
    lr_decay,
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
    print("-----------------------------")
    for ind, (input, target) in enumerate(train_loader):
        adjust_learning_rate(
            optimizer,
            epoch,
            ind,
            len(train_loader),
            total_epochs,
            args.lr,
            lr_decay,
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
            print(plot_progress)
        # if i == len(train_loader.dataloader) - 1:
        #     top1, _ = accuracy(output, target, topk=(1, 5))

    bar.finish()
    print("-----------------------------")
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
    bar.finish()
    return (losses.avg, top1.avg)


def adjust_learning_rate(
    optimizer,
    epoch,
    iteration,
    num_iter,
    epochs,
    lr_init,
    lr_decay,
    schedule=None,
    gamma=None,
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
