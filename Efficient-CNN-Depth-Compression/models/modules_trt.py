from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class NaiveFeed(nn.Module):
    def __init__(self, odict: OrderedDict) -> None:
        super().__init__()
        self.md = nn.Sequential(odict)

    def forward(self, x):
        return self.md(x)


class SkipFeed(nn.Module):
    def __init__(self, odict: OrderedDict, last=nn.Identity) -> None:
        super().__init__()
        self.md = nn.Sequential(odict)
        self.last = last()

    def forward(self, x):
        return self.last(self.md(x) + x)


class Downsample(nn.Module):
    def __init__(self, planes) -> None:
        super().__init__()
        self.planes = planes

    def forward(self, x):
        sz = x.shape[3] // 2
        ch = x.shape[1] // 2
        out = x
        out = F.interpolate(out, size=(sz, sz))
        zeros = out.mul(0)
        out = torch.cat((zeros[:, :ch, :, :], out), 1)
        out = torch.cat((out, zeros[:, ch:, :, :]), 1)
        return out


class SkipFeedDown(nn.Module):
    def __init__(
        self, odict: OrderedDict, last=nn.Identity, downsample=nn.Identity()
    ) -> None:
        super().__init__()
        self.md = nn.Sequential(odict)
        self.last = last()
        self.downsample = downsample

    def forward(self, x):
        return self.last(self.md(x) + self.downsample(x))
