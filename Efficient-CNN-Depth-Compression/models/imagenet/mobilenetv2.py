import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import torch
import torch.nn.functional as F
from functools import partial
from typing import Any, Optional, List, Tuple

from torch import Tensor
from torch import nn
from collections import OrderedDict

from torchvision.ops.misc import Conv2dNormActivation
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import (
    handle_legacy_interface,
    _ovewrite_named_param,
    _make_divisible,
)

from models.modules_trt import SkipFeed, NaiveFeed
from models.model_op import fuse_bn
from utils.measure import compile_and_time, torch_time, unroll_merged, torch_cpu_time
from fvcore.nn import FlopCountAnalysis

__all__ = ["InvertedResidual", "MobileNetV2", "MobileNet_V2_Weights", "mobilenet_v2"]


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer=nn.BatchNorm2d,
        activation_layer=partial(nn.ReLU6, inplace=True),
        last_relu=False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.activation_layer = activation_layer
        act_layers = self.make_act_layers(3 if last_relu else 2)
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[Tuple[str, torch.nn.Module]] = []

        ind = 1
        if expand_ratio != 1:
            layers += [
                (f"conv{ind}", nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False)),
                (f"bn{ind}", norm_layer(hidden_dim)),
                (f"relu{ind}", act_layers[0]),
            ]
            ind += 1

        layers += [
            (
                f"conv{ind}",
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
            ),
            (f"bn{ind}", norm_layer(hidden_dim)),
            (f"relu{ind}", act_layers[1]),
            (f"conv{ind+1}", nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
            (f"bn{ind+1}", norm_layer(oup)),
        ]

        if last_relu:
            layers += [(f"relu{ind+1}", act_layers[2])]

        # Each indicating if i-th act is disappeared
        self.is_merged = False
        self.m_layers: List[nn.Module] = []
        self.m_seq: nn.Sequential = None
        self.conv = nn.Sequential(OrderedDict(layers))
        self.in_channels = inp
        self.out_channels = oup
        self._is_cn = stride > 1

    def make_act_layers(self, num=2):
        return [self.activation_layer() for _ in range(num)]

    def forward(self, x: Tensor) -> Tensor:
        if self.is_merged:
            out = self.m_seq(x)
        else:
            pre_skip = len(self.conv)
            if isinstance(self.conv[-1], nn.ReLU6):
                pre_skip -= 1
            out = self.conv[:pre_skip](x)

            if self.use_res_connect:
                # In fine-tuning stage, the size might not match
                diff = (out.shape[-1] - x.shape[-1]) // 2
                x = F.pad(x, [diff] * 4)
                out = out + x

            out = self.conv[pre_skip:](out)
        return out


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block=InvertedResidual,
        norm_layer=nn.BatchNorm2d,
        dropout: float = 0.2,
        add_relu: bool = False,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()
        _log_api_usage_once(self)

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, isize
                [1, 16, 1, 1, 112],
                [6, 24, 2, 2, 112],
                [6, 32, 3, 2, 56],
                [6, 64, 4, 2, 28],
                [6, 96, 3, 1, 14],
                [6, 160, 3, 2, 14],
                [6, 320, 1, 1, 7],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 5
        ):
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 5-element list, got {inverted_residual_setting}"
            )
        self.add_relu = add_relu
        # building first layer
        self.norm_layer = norm_layer
        self.block = block
        self.width_mult = width_mult
        self.round_nearest = round_nearest
        self.dropout = dropout
        self.num_classes = num_classes

        self.build(input_channel, last_channel, inverted_residual_setting)
        self.initialize()

        self.is_merged = False
        self.is_fixed_act = False
        self.m_blocks: List[nn.Module] = []
        self.m_features = None  # nn.Sequential
        self.compress_k: int = 0

    def make_first_feature(self, input_channel):
        return Conv2dNormActivation(
            3,
            input_channel,
            stride=2,
            norm_layer=self.norm_layer,
            activation_layer=partial(nn.ReLU6, inplace=True),
        )

    def make_feature(self, input_channel, output_channel, stride, t):
        return self.block(
            input_channel,
            output_channel,
            stride,
            expand_ratio=t,
            norm_layer=self.norm_layer,
            last_relu=self.add_relu,
        )

    def make_last_feature(self, input_channel):
        return Conv2dNormActivation(
            input_channel,
            self.last_channel,
            kernel_size=1,
            norm_layer=self.norm_layer,
            activation_layer=partial(nn.ReLU6, inplace=True),
        )

    def set_classifier(self):
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.last_channel, self.num_classes),
        )

    def initialize(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def build(self, input_channel, last_channel, inverted_residual_setting):
        input_channel = _make_divisible(
            input_channel * self.width_mult, self.round_nearest
        )
        self.last_channel = _make_divisible(
            last_channel * max(1.0, self.width_mult), self.round_nearest
        )

        features: List[nn.Module] = []
        features.append(self.make_first_feature(input_channel))
        # building inverted residual blocks
        for t, c, n, s, isize in inverted_residual_setting:
            output_channel = _make_divisible(c * self.width_mult, self.round_nearest)
            self.cur_isize = isize
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    self.make_feature(input_channel, output_channel, stride, t)
                )
                self.cur_isize = self.cur_isize // stride
                input_channel = output_channel
        features.append(self.make_last_feature(input_channel))
        self.features = nn.Sequential(*features)
        self.set_classifier()

    def set_act_hats(self):
        pass

    def _forward_impl(self, x: Tensor) -> Tensor:
        if self.is_merged:
            x = self.m_features(x)
        else:
            if not self.is_fixed_act:
                self.set_act_hats()
            x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Does not merge anything and fuse batchnorm layer
    def merge(self):
        self.is_merged = True
        stack = []
        for _, block in enumerate(self.features):
            if isinstance(block, InvertedResidual):
                if block.use_res_connect:
                    stack.append(SkipFeed(block.conv._modules))
                else:
                    stack.append(NaiveFeed(block.conv._modules))
            else:
                stack.append(block)

        self.m_features = nn.Sequential(*stack)
        fuse_bn(self.m_features)
        print("Fused batchnorm...")
        delattr(self, "features")

    def time(self, txt="model", verb=False, trt=True):
        assert self.is_merged
        unrolled_module = unroll_merged(self)
        print(unrolled_module)

        if trt:
            print("Start compiling merged model...")
            result, std = compile_and_time(
                unrolled_module, (128, 3, 224, 224), txt, verb
            )
        else:
            result, std = torch_time(unrolled_module, (128, 3, 224, 224), txt, verb)
        del unrolled_module
        return result, std

    def flops(self):
        assert self.is_merged
        unrolled_module = unroll_merged(self)
        unrolled_module.eval()

        print(unrolled_module)
        inputs = torch.randn((1, 3, 224, 224)).cuda()
        flops = FlopCountAnalysis(unrolled_module, inputs)
        print(f"number of MFLOPs: {flops.total() / 1e6}")

        del unrolled_module
        return flops.total()

    def mem(self):
        assert self.is_merged
        mem = torch.cuda.max_memory_allocated()
        print(f"Before : {mem / 1e6:>15} MB")

        unrolled_module = unroll_merged(self)
        unrolled_module.eval()
        params = sum(p.numel() for p in unrolled_module.parameters())
        inputs = torch.randn((128, 3, 224, 224)).cuda()

        print(unrolled_module)
        for i in range(10):
            torch.cuda.reset_peak_memory_stats()
            unrolled_module(inputs)

            mem = torch.cuda.max_memory_allocated()
            print(f"Iter {i}  : {mem / 1e6:>15} MB")

        return params, mem

    def cpu_time(self, txt="model", verb=False):
        assert self.is_merged
        unrolled_module = unroll_merged(self)

        print(unrolled_module)

        result, std = torch_cpu_time(unrolled_module, (128, 3, 224, 224), txt, verb)
        del unrolled_module
        return result, std


_COMMON_META = {
    "num_params": 3504872,
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class MobileNet_V2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.878,
                    "acc@5": 90.286,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.154,
                    "acc@5": 90.822,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@handle_legacy_interface(weights=("pretrained", MobileNet_V2_Weights.IMAGENET1K_V1))
def mobilenet_v2(
    *,
    weights: Optional[MobileNet_V2_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> MobileNetV2:
    """MobileNetV2 architecture from the `MobileNetV2: Inverted Residuals and Linear
    Bottlenecks <https://arxiv.org/abs/1801.04381>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenetv2.MobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
    """
    weights = MobileNet_V2_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV2(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


# The dictionary below is internal implementation detail and will be removed in v0.15
from torchvision.models._utils import _ModelURLs

model_urls = _ModelURLs(
    {
        "mobilenet_v2": MobileNet_V2_Weights.IMAGENET1K_V1.url,
    }
)
