import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from functools import partial
from collections import OrderedDict
from models.imagenet.mobilenetv2 import InvertedResidual, MobileNetV2

from typing import Any, Optional, List
from models.model_op import (
    get_act,
    get_blk,
    adjust_padding,
    fix_act_lyrs,
    add_nonlinear,
    merge_or_new,
    push_merged_layers,
    fuse_skip,
    DepShrinkReLU6,
)
from models.modules_trt import SkipFeed, NaiveFeed

import torch
from torch import nn

from torchvision.models._meta import _IMAGENET_CATEGORIES

__all__ = ["DepShrinkMobileNetV2", "dep_shrink_mobilenet_v2"]


class DepShrinkInvertedResidual(InvertedResidual):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        isize: int,
        norm_layer=nn.BatchNorm2d,
    ) -> None:
        super().__init__(
            inp=inp,
            oup=oup,
            stride=stride,
            expand_ratio=expand_ratio,
            norm_layer=norm_layer,
            activation_layer=DepShrinkReLU6,
        )
        self.isize = isize

    def make_act_layers(self, _):
        self.act_layer = self.activation_layer()
        return self.act_layer, self.act_layer

    def set_act_hat(self, val):
        self.act_layer.act_hat = val

    def get_act_hat(self):
        return self.act_layer.act_hat

    def merge(self, merged):
        if self.is_merged:
            print("Model already Merged")
            return
        self.is_merged = True

        # weights = (isize, m_cw, m_p, m_b), cparam = (ctype, cstr)
        weights, cparam = (self.isize, None, None, None), ("dw", 1)
        pos = 0

        last_relu = isinstance(self.conv[-1], nn.ReLU6)
        for ind in range(0, len(self.conv), 3):

            is_merged = False if ind == 0 else merged
            is_last = ind >= len(self.conv) - 3

            relu = not (ind == len(self.conv) - 2)

            conv, bn = self.conv[ind], self.conv[ind + 1]
            lyrs = (conv, bn)

            pos, weights, cparam = merge_or_new(
                self.m_layers, pos, weights, lyrs, is_merged, cparam
            )
            if all([merged, is_last, self.use_res_connect]):
                fuse_skip(weights[1])
                self.use_res_connect = False

            push_merged_layers(self.m_layers, pos, weights, cparam, relu=relu)

        delattr(self, "conv")

        if self.use_res_connect and last_relu:
            layer = partial(nn.ReLU6, inplace=True)
            self.m_seq = SkipFeed(OrderedDict(self.m_layers[:-1]), layer)
        elif self.use_res_connect and not last_relu:
            self.m_seq = SkipFeed(OrderedDict(self.m_layers))
        else:
            self.m_seq = NaiveFeed(OrderedDict(self.m_layers))

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        if self.use_res_connect:
            out = f"[Skip : Enabled]"
        else:
            out = f"[Skip : Disabled]"
        return out


class DepShrinkMobileNetV2(MobileNetV2):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block=DepShrinkInvertedResidual,
        norm_layer=nn.BatchNorm2d,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            num_classes,
            width_mult,
            inverted_residual_setting,
            round_nearest,
            block,
            norm_layer,
            dropout,
        )

        self.compress_k = -1
        self.act_pos = set()

    def make_feature(self, input_channel, output_channel, stride, t):
        return self.block(
            input_channel,
            output_channel,
            stride,
            expand_ratio=t,
            isize=self.cur_isize,
            norm_layer=self.norm_layer,
        )

    def load_pattern(self, pat):
        if pat == "none":
            return
        elif pat == "A":
            lst = [0, 3, 4, 6, 8, 10, 11, 13, 14, 15, 16]
        elif pat == "B":
            lst = [3, 4, 10, 11, 13, 14, 15, 16]
        elif pat == "C":
            lst = [0, 3, 10, 11, 13, 14, 15, 16]
        elif pat == "D":
            lst = [3, 4, 13, 14, 15, 16]
        elif pat == "E":
            lst = [0, 3, 13, 14, 15, 16]
        elif pat == "F":
            lst = []
        elif pat == "A10":
            lst = [2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
        elif pat == "B10":
            lst = [0, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
        elif pat == "C10":
            lst = [0, 3, 4, 6, 10, 13, 14, 15, 16]
        elif pat == "D10":
            lst = [0, 3, 10, 11, 13, 15, 16]
        elif pat == "AR":
            lst = [5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        elif pat == "BR":
            lst = [4, 10, 11, 12, 13, 14, 15, 16]
        elif pat == "CR":
            lst = [8, 9, 10, 11, 12, 16]
        elif pat == "AR10":
            lst = [4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        elif pat == "BR10":
            lst = [4, 7, 8, 9, 12, 13, 14, 15, 16]
        elif pat == "CR10":
            lst = [8, 10, 12, 13, 14, 15, 16]
        elif pat == "AR_AUG":
            lst = [0, 3, 5, 7, 8, 9, 12, 13, 14, 15, 16]
        elif pat == "BR_AUG":
            lst = [5, 7, 8, 9, 12, 14, 15, 16]
        elif pat == "CR_AUG":
            lst = [9, 11, 12, 14, 15, 16]
        elif pat == "AR10_AUG":
            lst = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        elif pat == "BR10_AUG":
            lst = [5, 7, 8, 9, 12, 13, 14, 15, 16]
        elif pat == "CR10_AUG":
            lst = [6, 7, 9, 13, 14, 15, 16]
        else:
            raise NotImplementedError()

        for ind, block in enumerate(self.features[1:-1]):
            if ind in lst:
                block.act_layer.act.data = torch.tensor([1e5])
                block.act_layer.act.requires_grad = False
            else:
                block.act_layer.act.data = torch.tensor([-1e5])
                block.act_layer.act.requires_grad = False

        print(f"Loaded pattern {pat}")

    def set_act_hats(self):
        assert self.compress_k >= 0
        lst, k = [], self.compress_k
        for block in self.features[1:-1]:
            if isinstance(block, DepShrinkInvertedResidual):
                lst.append(block.act_layer.act.data)
            else:
                lst.append(torch.tensor([float("-inf")]))

        acts = torch.cat(lst)
        _, act_ind = torch.sort(acts, stable=True)

        # set act_hats for top-k blocks
        for ind, block in enumerate(self.features[1:-1]):
            if isinstance(block, DepShrinkInvertedResidual):
                if k == 0:
                    block.set_act_hat(0.0)
                elif ind in act_ind[-k:]:
                    block.set_act_hat(1.0)
                else:
                    block.set_act_hat(0.0)

    def get_act_info(self):
        self.set_act_hats()
        act_pos, act_num = get_act(self.features[1:-1], InvertedResidual)
        # DepthShrinker appends activation to merged blocks
        blk_pos = get_blk(self.features[1:-1], InvertedResidual)
        add_pos, blk_empty = set(), True
        for i in range(act_num + 1):
            if i in act_pos:
                blk_empty = False
            if i in blk_pos:
                if blk_empty:
                    add_pos.add(i)
                else:
                    blk_empty = True
        act_pos = set.union(act_pos, add_pos)
        return act_pos, act_num

    def get_arch_parameters(self):
        return [w for name, w in self.named_parameters() if "act" in name]

    def regularizer(self, mode="soft"):
        if mode == "soft":
            arch_params = torch.cat(self.get_arch_parameters())
            loss = torch.sum(arch_params)
        elif mode == "w1.4":
            # fmt: off
            lat = torch.tensor([15.19, 133.34, 75.77, 43.8, 30.39, 30.39, 18.81, 15.19, 15.19, 15.19, 17.0, 26.23, 26.23, 13.23, 11.41, 11.41, 14.39])
            # fmt: on
            arch_params = torch.cat(self.get_arch_parameters())
            assert len(lat) == len(arch_params)
            loss = torch.sum(torch.mul(lat.to("cuda"), arch_params))
        elif mode == "w1.0":
            # fmt: off
            lat = torch.tensor([18.15, 85.52, 57.97, 35.78, 20.93, 20.93, 12.51, 11.45, 11.45, 11.45, 11.64, 17.85, 17.85, 9.49, 7.35, 7.35, 8.48])
            # fmt: on
            arch_params = torch.cat(self.get_arch_parameters())
            assert len(lat) == len(arch_params)
            loss = torch.sum(torch.mul(lat.to("cuda"), arch_params))
        else:
            raise NotImplementedError()
        return loss

    def fix_act(self, act_pos=None):
        if self.is_fixed_act:
            print("Model already Fixed")
            return
        self.is_fixed_act = True

        # position in nodes and bumps denote the number of come-acrossed conv
        self.act_pos = act_pos if act_pos != None else self.get_act_info()[0]
        blk_end = sorted(get_blk(self.features[1:-1], InvertedResidual))

        fix_act_lyrs(self.features[1:-1], InvertedResidual, self.act_pos)
        add_nonlinear(self.features[1:-1], self.act_pos)
        adjust_padding(self.features[1:-1], set.union(self.act_pos, blk_end))

        for block in self.features[1:-1]:
            delattr(block, "act_layer")

    def merge(self, act_pos=None):
        if self.is_merged:
            print("Model already Merged")
            return
        self.is_merged = True

        # position in nodes and bumps denote the number of come-acrossed conv
        self.act_pos = act_pos if act_pos != None else self.get_act_info()[0]

        node_pos = 0
        for _, block in enumerate(self.features[1:-1]):
            for _, layer in block.conv._modules.items():
                if isinstance(layer, nn.Conv2d):
                    node_pos += 1
            self.m_blocks.append(block)
            is_merged = node_pos in self.act_pos
            if isinstance(block, DepShrinkInvertedResidual):
                block.merge(is_merged)

        self.m_blocks = [self.features[0]] + self.m_blocks + [self.features[-1]]

        stack = []
        for _, block in enumerate(self.m_blocks):
            if isinstance(block, DepShrinkInvertedResidual):
                if block.use_res_connect:
                    stack.append(
                        SkipFeed(block.m_seq.md._modules, block.m_seq.last.__class__)
                    )
                else:
                    stack.append(NaiveFeed(block.m_seq.md._modules))
            else:
                stack.append(block)

        self.m_features = nn.Sequential(*stack)

        delattr(self, "features")


_COMMON_META = {
    "num_params": 3504872,
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


def dep_shrink_mobilenet_v2(
    **kwargs: Any,
) -> DepShrinkMobileNetV2:
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

    model = DepShrinkMobileNetV2(**kwargs)

    return model
