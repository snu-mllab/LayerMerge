import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import utils.datasets as datasets

from timm.data import create_transform


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
