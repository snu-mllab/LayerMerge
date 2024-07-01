import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from typing import Any, Optional, List
from collections import OrderedDict
from functools import partial
from models.model_op import (
    adjust_padding,
    adjust_isize,
    merge_or_new,
    push_merged_layers,
    trace_feat_size,
    get_act,
    get_blk,
    fix_act_lyrs,
)
from models.imagenet.vgg import VGG, VGGBlock, vgg_cfgs
from models.modules_trt import NaiveFeed, SkipFeed
from utils.measure import compile_and_time

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._meta import _IMAGENET_CATEGORIES

__all__ = ["LearnVGG", "LearnVGGBlock", "learn_vgg19_bn"]


def push_mp_layer(stack, new_name):
    layer = nn.MaxPool2d(kernel_size=2, stride=2)
    if isinstance(stack, list):
        stack.append((new_name, layer))
    elif isinstance(stack, nn.Sequential):
        stack.add_module(new_name, layer)
    else:
        raise NotImplementedError("Not implemented type of stack")


class LearnVGGBlock(VGGBlock):
    def __init__(
        self,
        inp: int,
        oup: int,
        num: int,
        isize: int,
    ) -> None:
        super().__init__(inp, oup, num)
        self.isize = isize

    def extra_repr(self):
        out = f"[isize : {self.isize}]"
        return out


class LearnVGG(VGG):
    def __init__(
        self,
        cfg,
        num_classes: int = 1000,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(cfg, num_classes, dropout, LearnVGGBlock)

        self.act_pos = set()
        self.str_pos = {2, 4, 8, 12, 16}
        # No skip-connections
        self.skip_s2t = dict()

        inp = torch.randn(64, 3, 224, 224)
        self.out_shape = trace_feat_size(self.features, inp)

    def get_act_info(self):
        act_pos, act_num = get_act(self.features, VGGBlock)
        return act_pos, act_num

    def get_blk_info(self):
        blk_end = sorted(get_blk(self.features, VGGBlock))
        blks = dict([(ind + 1, end) for ind, end in enumerate(blk_end)])
        blk_num = len(blks)
        return blks, blk_num

    def make_feature(self, inp, oup, n):
        return self.block(inp, oup, n, self.cur_isize)

    def fix_act(self, act_pos=None, merge_pos=None):
        self.is_fixed_act = True

        self.act_pos = act_pos if act_pos != None else self.get_act_info()[0]
        self.merge_pos = merge_pos if merge_pos != None else self.act_pos
        assert self.act_pos.issubset(self.merge_pos)

        bumps = set.union(self.str_pos, self.merge_pos)

        fix_act_lyrs(self.features, VGGBlock, self.act_pos, "relu")
        adjust_padding(self.features, bumps)
        adjust_isize(self.features)

    def merge(self, act_pos=None, merge_pos=None, keep_feat=False):
        self.to("cpu")
        self.is_merged = True

        self.act_pos = act_pos if act_pos != None else self.get_act_info()[0]
        self.merge_pos = merge_pos if merge_pos != None else self.act_pos
        assert self.act_pos.issubset(self.merge_pos)

        bumps = set.union(self.str_pos, self.merge_pos)

        self.m_layers, stack = [], []
        pos, bump_end, loc_end, last = 0, None, None, None

        # weights = (isize, m_cw, m_p, m_b), cparam = (ctype, cstr)
        weights, cparam = (112, None, None, None), ("dw", 1)

        for block in self.features:
            for ind in range(0, len(block.conv) - 1, 3):
                relu = pos + 1 in self.act_pos
                conv, bn = block.conv[ind], block.conv[ind + 1]
                lyrs = (conv, bn)

                is_merged = pos > 0 and not pos in bumps

                _, weights, cparam = merge_or_new(
                    stack, pos, weights, lyrs, is_merged, cparam, last
                )
                push_merged_layers(
                    stack, pos, weights, cparam, relu=relu, act_type="relu"
                )
                pos += 1
                last = 2 if relu else 1
            push_mp_layer(stack, f"pool{pos-1}")

        if len(stack) > 0:
            self.m_layers += [NaiveFeed(OrderedDict(stack))]
        self.m_features = nn.Sequential(*self.m_layers)

        if not keep_feat:
            delattr(self, "features")
        self.to("cuda")


def learn_vgg19_bn(num_classes):
    return LearnVGG(vgg_cfgs["19"], num_classes=num_classes)
