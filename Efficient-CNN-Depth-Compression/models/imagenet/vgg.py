from functools import partial
from typing import Union, List, Dict, Any, Optional, cast
from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param

from models.modules_trt import NaiveFeed
from models.model_op import fuse_bn
from utils.measure import compile_and_time, torch_time, unroll_merged_vgg

__all__ = [
    "VGG",
    "VGGBlock",
    "vgg19_bn",
    "vgg_cfgs",
]


class VGGBlock(nn.Module):
    def __init__(self, inp: int, oup: int, num: int) -> None:
        super().__init__()
        layers = []
        ind = 1
        for _ in range(num):
            if ind != 1:
                inp = oup
            layers += [
                (f"conv{ind}", nn.Conv2d(inp, oup, kernel_size=3, padding=1)),
                (f"bn{ind}", nn.BatchNorm2d(oup)),
                (f"relu{ind}", nn.ReLU(inplace=True)),
            ]
            ind += 1
        layers += [(f"pool{ind}", nn.MaxPool2d(kernel_size=2, stride=2))]

        # Each indicating if i-th act is disappeared
        self.conv = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        out = self.conv(x)
        return out


class VGG(nn.Module):
    def __init__(
        self,
        cfg,
        num_classes: int = 1000,
        dropout: float = 0.5,
        block: nn.Module = VGGBlock,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.dropout = dropout
        self.num_classes = num_classes
        self.block = block

        self.build(cfg)
        self.initialize()

        self.is_merged = False

    def set_classifier(self):
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, self.num_classes),
        )

    def make_feature(self, inp, oup, n):
        return self.block(inp, oup, n)

    def build(self, cfg):
        features = []
        for n, inp, oup, isize in cfg:
            self.cur_isize = isize
            features.append(self.make_feature(inp, oup, n))
        self.features = nn.Sequential(*features)
        self.set_classifier()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_merged:
            x = self.m_features(x)
        else:
            x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # Does not merge anything and fuse batchnorm layer
    def merge(self):
        self.is_merged = True
        stack = []
        for _, block in enumerate(self.features):
            stack.append(NaiveFeed(block.conv._modules))

        self.m_features = nn.Sequential(*stack)
        fuse_bn(self.m_features)
        print("Fused batchnorm...")
        delattr(self, "features")

    def time(self, txt="model", verb=False, trt=True):
        assert self.is_merged
        unrolled_module = unroll_merged_vgg(self)
        print(unrolled_module)

        if trt:
            print("Start compiling merged model...")
            result, std = compile_and_time(
                unrolled_module, (64, 3, 224, 224), txt, verb
            )
        else:
            result, std = torch_time(
                unrolled_module, (64, 3, 224, 224), txt, verb, rep=200, warmup=300
            )
        del unrolled_module
        return result, std

    def mem(self):
        assert self.is_merged
        mem = torch.cuda.max_memory_allocated()
        print(f"Before : {mem / 1e6:>15} MB")

        unrolled_module = unroll_merged_vgg(self)
        unrolled_module.eval()
        params = sum(p.numel() for p in unrolled_module.parameters())
        inputs = torch.randn((64, 3, 224, 224)).cuda()

        print(unrolled_module)
        for i in range(10):
            torch.cuda.reset_peak_memory_stats()
            unrolled_module(inputs)

            mem = torch.cuda.max_memory_allocated()
            print(f"Iter {i}  : {mem / 1e6:>15} MB")

        return params, mem


# fmt: off
# cfgs: Dict[str, List[Union[str, int]]] = {
#     "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
#     "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
# }
# fmt: on

vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    # (n, inp, oup, isize)
    "19": [
        (2, 3, 64, 224),
        (2, 64, 128, 112),
        (4, 128, 256, 56),
        (4, 256, 512, 28),
        (4, 512, 512, 14),
    ]
}


def _vgg(
    cfg: str,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(
                kwargs, "num_classes", len(weights.meta["categories"])
            )
    model = VGG(vgg_cfgs[cfg], **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def vgg19_bn(*, weights=None, progress=True, **kwargs: Any) -> VGG:
    return _vgg("19", weights, progress, **kwargs)
