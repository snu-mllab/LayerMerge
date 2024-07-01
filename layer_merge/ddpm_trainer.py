import time
import torch
import numpy as np

from tqdm import tqdm
from progress.bar import Bar as Bar
from layer_merge.models.ddpm_datasets import data_transform

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

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
        

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def ddpm_train(train_loader, model, optimizer):
    bar = Bar("Processing", max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    print("-----------------------------")
    for i, (x, y) in enumerate(train_loader):
        n = x.size(0)
        data_time.update(time.time() - end)
        model.train()

        x = x.to("cuda")
        x = data_transform(model.config, x)
        e = torch.randn_like(x)
        b = get_beta_schedule(
                beta_schedule=model.config.diffusion.beta_schedule,
                beta_start=model.config.diffusion.beta_start,
                beta_end=model.config.diffusion.beta_end,
                num_diffusion_timesteps=model.config.diffusion.num_diffusion_timesteps,
            )
        b = torch.from_numpy(b).float().to("cuda")

        # antithetic sampling
        t = torch.randint(
            low=0, high=1000, size=(n // 2 + 1,)
        ).to("cuda")
        t = torch.cat([t, 1000 - t - 1], dim=0)[:n]
        loss = noise_estimation_loss(model, x, t, e, b)

        losses.update(loss.item(), x.size(0))
        #tb_logger.add_scalar("loss", loss, global_step=step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} ".format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
        )
        bar.next()
    bar.finish()
    print("-----------------------------")
    return losses.avg


def ddpm_validate(val_loader, model):
    bar = Bar("Processing", max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (x, _) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x = x.to("cuda")
        x = data_transform(model.config, x)
        n = x.size(0)
        e = torch.randn_like(x)
        b = get_beta_schedule(
                beta_schedule=model.config.diffusion.beta_schedule,
                beta_start=model.config.diffusion.beta_start,
                beta_end=model.config.diffusion.beta_end,
                num_diffusion_timesteps=model.config.diffusion.num_diffusion_timesteps,
            )
        b = torch.from_numpy(b).float().to("cuda")
        
        loss = 0
        with torch.no_grad():
            # antithetic sampling
            t = torch.randint(
                low=0, high=1000, size=(n // 2 + 1,)
            ).to("cuda")
            t = torch.cat([t, 1000 - t - 1], dim=0)[:n]
            loss = noise_estimation_loss(model, x, t, e, b)
        print(loss)

        losses.update(loss.item(), x.size(0))
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

    return losses.avg
