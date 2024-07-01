import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from typing import Any, Optional, List
from collections import OrderedDict
from functools import partial
from models.model_op import (
    get_skip_info,
    get_skip_bumps,
    adjust_padding,
    adjust_isize,
    merge_or_new,
    push_merged_layers,
    fuse_skip,
    trace_feat_size,
    get_act,
    get_blk,
    fix_act_lyrs,
)
from models.imagenet.mobilenetv2 import InvertedResidual, MobileNetV2
from models.modules_trt import NaiveFeed, SkipFeed

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._meta import _IMAGENET_CATEGORIES

__all__ = ["LearnMobileNetV2", "learn_mobilenet_v2"]


def get_merged_list(module):
    merged = []
    if isinstance(module, LearnInvertedResidual):
        for _, act_ind in enumerate(range(2, len(module.conv), 3)):
            act_lyr = module.conv[act_ind]
            if isinstance(act_lyr, (nn.Identity, nn.ReLU6)):
                merged += [isinstance(act_lyr, nn.Identity)]
    else:
        raise NotImplementedError()
    return merged


class LearnInvertedResidual(InvertedResidual):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        isize: int,
        norm_layer=nn.BatchNorm2d,
        last_relu=False,
    ) -> None:
        super().__init__(
            inp=inp,
            oup=oup,
            stride=stride,
            expand_ratio=expand_ratio,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU6,
            last_relu=last_relu,
        )
        self.isize = isize
        self.merged = [False] * (len(self.conv) // 3)

    def merge(self):
        if self.is_merged:
            print("Model already Merged")
            return
        self.is_merged = True

        # weights = (isize, m_cw, m_p, m_b), cparam = (ctype, cstr)
        weights, cparam = (self.isize, None, None, None), ("dw", 1)
        pos = 0

        # Bring merged list from act_lyr
        self.merged = get_merged_list(self)

        for ind in range(0, len(self.conv), 3):

            is_merged = False if ind == 0 else self.merged[(ind - 2) // 3]
            is_last = ind >= len(self.conv) - 3

            relu = not (ind == len(self.conv) - 2)
            if (ind == len(self.conv) - 3) and self.merged[-1]:
                relu = False

            conv, bn = self.conv[ind], self.conv[ind + 1]
            lyrs = (conv, bn)

            pos, weights, cparam = merge_or_new(
                self.m_layers, pos, weights, lyrs, is_merged, cparam
            )
            if all(self.merged + [is_last, self.use_res_connect]):
                fuse_skip(weights[1])
                self.use_res_connect = False

            push_merged_layers(self.m_layers, pos, weights, cparam, relu=relu)

        delattr(self, "conv")

        if not self.use_res_connect:
            self.m_seq = NaiveFeed(OrderedDict(self.m_layers))
        elif relu:
            self.m_seq = SkipFeed(OrderedDict(self.m_layers[:-1]), nn.ReLU6)
        elif not relu:
            self.m_seq = SkipFeed(OrderedDict(self.m_layers))

    def fix_act(self):
        self.conv.conv1.padding = (1, 1)
        self.conv.conv2.padding = (0, 0)
        if hasattr(self.conv, "conv3"):
            self.conv.conv3.padding = (0, 0)

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        if self.use_res_connect:
            out = f"[Skip : Enabled]  [isize : {self.isize}]"
        else:
            out = f"[Skip : Disabled] [isize : {self.isize}]"
        return out


class LearnMobileNetV2(MobileNetV2):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block=LearnInvertedResidual,
        norm_layer=nn.BatchNorm2d,
        dropout: float = 0.2,
        add_relu: bool = False,
    ) -> None:

        super().__init__(
            num_classes,
            width_mult,
            inverted_residual_setting,
            round_nearest,
            block,
            norm_layer,
            dropout,
            add_relu,
        )

        self.skip_s2t, self.skip_t2s, self.str_pos = get_skip_info(self.features[1:-1])
        self.skip_bump, self.act_pos = set(), set()

        inp = torch.randn(128, 3, 224, 224)
        out = self.features[0](inp)
        self.out_shape = trace_feat_size(self.features[1:-1], out)

    def make_feature(self, input_channel, output_channel, stride, t):
        return self.block(
            input_channel,
            output_channel,
            stride,
            expand_ratio=t,
            isize=self.cur_isize,
            norm_layer=self.norm_layer,
            last_relu=self.add_relu,
        )

    def get_act_info(self):
        act_pos, act_num = get_act(self.features[1:-1], InvertedResidual)
        return act_pos, act_num

    def get_blk_info(self):
        blk_end = sorted(get_blk(self.features[1:-1], InvertedResidual))
        blks = dict([(ind + 1, end) for ind, end in enumerate(blk_end)])
        blk_num = len(blks)
        return blks, blk_num

    def fix_act(self, act_pos=None, merge_pos=None):
        self.is_fixed_act = True

        # position in nodes and bumps denote the number of come-acrossed conv
        self.act_pos = act_pos if act_pos != None else self.get_act_info()[0]
        self.merge_pos = merge_pos if merge_pos != None else self.act_pos
        assert self.act_pos.issubset(self.merge_pos)

        self.skip_bump, _ = get_skip_bumps(self.merge_pos, self.skip_s2t)
        bumps = set.union(self.str_pos, self.merge_pos, self.skip_bump)

        fix_act_lyrs(self.features[1:-1], InvertedResidual, self.act_pos)
        # adjust padding and input size among the conv layers w.r.t. bumps
        adjust_padding(self.features[1:-1], bumps)
        adjust_isize(self.features[1:-1])

    def merge(self, act_pos=None, merge_pos=None, keep_feat=False):
        self.to("cpu")
        self.is_merged = True

        self.act_pos = act_pos if act_pos != None else self.get_act_info()[0]
        self.merge_pos = merge_pos if merge_pos != None else self.act_pos
        assert self.act_pos.issubset(self.merge_pos)

        self.skip_bump, skip_bump_s2t = get_skip_bumps(self.merge_pos, self.skip_s2t)
        bumps = set.union(self.str_pos, self.merge_pos, self.skip_bump)

        self.m_layers, stack = [], []
        pos, bump_end, loc_end, last = 0, None, None, None

        # weights = (isize, m_cw, m_p, m_b), cparam = (ctype, cstr)
        weights, cparam = (112, None, None, None), ("dw", 1)

        for block in self.features[1:-1]:
            for ind in range(0, len(block.conv), 3):

                if pos in skip_bump_s2t:
                    if len(stack) > 0:
                        self.m_layers += [NaiveFeed(OrderedDict(stack))]
                        stack = []
                    bump_end = skip_bump_s2t[pos]

                relu = not (ind == len(block.conv) - 2)
                if not pos + 1 in self.act_pos:
                    relu = False

                conv, bn = block.conv[ind], block.conv[ind + 1]
                lyrs = (conv, bn)

                is_merged = pos > 0 and not pos in bumps

                if pos in self.skip_s2t and not pos in skip_bump_s2t:
                    loc_end = self.skip_s2t[pos]
                    saved = (stack, weights, cparam, last, is_merged)
                    weights, cparam = (weights[0], None, None, None), ("dw", 1)
                    stack, last, is_merged = [], None, False

                if pos + 1 == loc_end:
                    _, weights, cparam = merge_or_new(
                        stack, pos, weights, lyrs, is_merged, cparam, last
                    )
                    fuse_skip(weights[1])
                    push_merged_layers(stack, pos, weights, cparam, relu=relu)

                    # Doesn't have bn (already fused), thus give `None` in the place of `bn`
                    lyrs = stack[0][1], None
                    relu = len(stack) > 1
                    stack, weights, cparam, last, is_merged = saved

                _, weights, cparam = merge_or_new(
                    stack, pos, weights, lyrs, is_merged, cparam, last
                )
                push_merged_layers(stack, pos, weights, cparam, relu=relu)

                if pos + 1 == bump_end:
                    if relu:
                        layer = partial(nn.ReLU6, inplace=True)
                        skipfeed = SkipFeed(OrderedDict(stack[:-1]), layer)
                    else:
                        skipfeed = SkipFeed(OrderedDict(stack))
                    self.m_layers += [skipfeed]
                    stack = []

                pos += 1
                last = 2 if relu else 1

        if len(stack) > 0:
            self.m_layers += [NaiveFeed(OrderedDict(stack))]
        self.m_layers = [self.features[0]] + self.m_layers + [self.features[-1]]
        self.m_features = nn.Sequential(*self.m_layers)

        if not keep_feat:
            delattr(self, "features")
        self.to("cuda")

    def unmerge(self):
        assert hasattr(self, "features")
        if self.is_merged:
            self.is_merged = False
            delattr(self, "m_features")


_COMMON_META = {
    "num_params": 3504872,
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


def learn_mobilenet_v2(
    **kwargs: Any,
) -> LearnMobileNetV2:
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
    model = LearnMobileNetV2(**kwargs)
    return model
