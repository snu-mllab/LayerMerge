from layer_merge.models.mobilenetv2 import InvertedResidual, MobileNetV2
from layer_merge.models.merge_op import NaiveFeed, SkipFeed, SkipFeedDown
from layer_merge.models.merge_op import (
    merge_or_new,
    fuse_skip,
    push_merged_layers,
    identity_conv_bn,
)

from functools import reduce, partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthInvertedResidual(InvertedResidual):
    def morph(self):
        layers = dict()
        if len(self.conv) == 3:
            layers["conv1"] = self.conv[0][0]
            layers["bn1"] = self.conv[0][1]
            layers["relu1"] = nn.ReLU6(inplace=True)
            layers["conv2"] = self.conv[1]
            layers["bn2"] = self.conv[2]
            layers["relu2"] = nn.Identity()
        elif len(self.conv) == 4:
            layers["conv1"] = self.conv[0][0]
            layers["bn1"] = self.conv[0][1]
            layers["relu1"] = nn.ReLU6(inplace=True)
            layers["conv2"] = self.conv[1][0]
            layers["bn2"] = self.conv[1][1]
            layers["relu2"] = nn.ReLU6(inplace=True)
            layers["conv3"] = self.conv[2]
            layers["bn3"] = self.conv[3]
            layers["relu3"] = nn.Identity()
        else:
            raise NotImplementedError()
        
        self.conv = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
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
    

class DepthLayerMobileNetV2(MobileNetV2):
    def morph(self):
        # layers
        self.cur_index = -1
        self.convs, self.bns, self.acts = {}, {}, {}
        self.outs = {}
        # network config
        self.skip_s2t = {}
        self.iden_pos = set()
        self.depth = -1
        for module in self.modules():
            if isinstance(module, InvertedResidual):
                module.__class__ = DepthInvertedResidual
                module.morph()
        self.act_pos = set()
        self.str_pos = set()

    def trace(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                self.cur_index += 1
                self.convs[self.cur_index] = (name, module)
            if isinstance(module, nn.BatchNorm2d):
                self.bns[self.cur_index] = (name, module)
            if isinstance(module, (nn.ReLU6, nn.Identity)):
                self.acts[self.cur_index] = (name, module)

        # Do not consider merging the last conv in mbv2 (kim23efficient)
        self.depth = len(self.acts) - 1

        if self.depth == 51:
            self.skip_s2t = {0: 2}
            self.skip_s2t = {i: i + 3 for i in range(2, 50, 3)}
            for ind in [2, 8, 17, 29, 38, 47]:
                del self.skip_s2t[ind]
            # original stride position:  {0, 4, 10, 19, 40}
            # 
            # In MobileNet-V2, we consider merging 1x1 conv
            # right after the stride-2 conv (kim23efficient)
            self.str_pos = {0, 5, 11, 20, 41}
            self.iden_pos = {i for i in range(2, 53, 3)}
        else:
            raise NotImplementedError()

        inp = torch.rand(128, 3, 224, 224)
        for ind, (_, conv) in self.convs.items():
            inp = conv(inp)
            self.outs[ind] = tuple(inp.shape)

    def set_name(self, name, object):
        name_lst = name.split(".")
        if len(name_lst) == 1:
            parent = self
        else:
            parent = reduce(getattr, name_lst[:-1], self)
        setattr(parent, name_lst[-1], object)
    
    def get_name(self, name):
        name_lst = name.split(".")
        if len(name_lst) == 1:
            parent = self
        else:
            parent = reduce(getattr, name_lst[:-1], self)
        return getattr(parent, name_lst[-1])
    
    def fix_act(self, act_pos: set):
        default_ind = set(range(self.depth)) - self.iden_pos

        nonlinear_act_pos = set.intersection(act_pos, default_ind)
        bf_ind = 0
        for ind in sorted(list(act_pos)):
            if ind > bf_ind + 1:
                nonlinear_act_pos.add(ind)
            bf_ind = ind

        self.act_pos = set(act_pos)
        assert self.act_pos.issubset(set(range(self.depth)))
        for ind, (act_name, _) in self.acts.items():
            # Do not consider merging the last conv in mbv2 (kim23efficient)
            if ind in set.union(nonlinear_act_pos, {self.depth}):
                self.set_name(act_name, nn.ReLU6(inplace=True))
            else:
                self.set_name(act_name, nn.Identity())
        new_acts = dict()
        for ind, (act_name, _) in self.acts.items():
            new_acts[ind] = (act_name, self.get_name(act_name))
        self.acts = new_acts
    
    def fix_conv(self, conv_pos: set):
        self.conv_pos = set(conv_pos)
        assert self.conv_pos.issubset(set(range(self.depth + 1)))

        new_convs, new_bns = dict(), dict()
        for ind, (conv_name, _) in self.convs.items():
            if not ind in set.union(self.conv_pos, {self.depth}):
                self.set_name(conv_name, nn.Identity())
            new_convs[ind] = (conv_name, self.get_name(conv_name))
        for ind, (bn_name, _) in self.bns.items():
            if not ind in set.union(self.conv_pos, {self.depth}):
                self.set_name(bn_name, nn.Identity())
            new_bns[ind] = (bn_name, self.get_name(bn_name))

        self.convs = new_convs
        self.bns = new_bns

    def get_skip_bumps(self, stop_pos):
        ind, stop_bumps = 0, sorted(list(stop_pos))
        skip_bumps, skip_bumps_s2t, l = set(), dict(), len(stop_bumps)
        if l == 0:
            return skip_bumps
        for src, tgt in self.skip_s2t.items():
            while stop_bumps[ind] < tgt:
                if stop_bumps[ind] > src:
                    skip_bumps.update([src, tgt])
                    skip_bumps_s2t[src] = tgt
                    break
                ind += 1
                if ind >= l:
                    return skip_bumps, skip_bumps_s2t
        return skip_bumps, skip_bumps_s2t

    def adjust_padding(self, merge_pos=None):
        pad, starting_layer = 0, None

        self.merge_pos = merge_pos if merge_pos != None else self.act_pos
        assert self.act_pos.issubset(self.merge_pos)
        assert self.conv_pos.issubset(set(range(self.depth)))

        stop_bumps = set.union(self.merge_pos, self.str_pos)
        skip_bumps, _ = self.get_skip_bumps(stop_bumps)
        bumps = set.union(skip_bumps, stop_bumps)
        for pos, (_, layer) in self.convs.items():
            if pos == 0:
                continue
            node_pos = pos - 1
            if starting_layer == None:
                starting_layer = layer
            is_start_iden = isinstance(starting_layer, nn.Identity)
            if is_start_iden:
                starting_layer = layer

            if node_pos in bumps and not node_pos == 0 and not is_start_iden:
                starting_layer.padding = (pad, pad)
                pad = 0
                starting_layer = layer
            if isinstance(layer, nn.Identity):
                continue
            pad += layer.padding[0]
            layer.padding = (0, 0)
        starting_layer.padding = (pad, pad)


    def merge(self, merge_pos=None):
        self.to("cpu")
        self.is_merged = True

        self.merge_pos = merge_pos if merge_pos != None else self.act_pos
        assert self.act_pos.issubset(self.merge_pos)

        stop_bumps = set.union(self.merge_pos, self.str_pos)
        skip_bumps, skip_bumps_s2t = self.get_skip_bumps(stop_bumps)
        bumps = set.union(skip_bumps, stop_bumps)

        self.m_layers, stack = [], []
        pos, bump_end, loc_end, last = -1, None, None, None

        # weights = (isize, m_cw, m_p, m_b), cparam = (ctype, cstr)
        weights, cparam = (112, None, None, None), ("dw", 1)

        for (_, conv), (_, bn), (_, act) in zip(
            self.convs.values(), self.bns.values(), self.acts.values()
        ):
            if pos in skip_bumps_s2t:
                if len(stack) > 0:
                    self.m_layers += [NaiveFeed(OrderedDict(stack))]
                    stack = []
                bump_end = skip_bumps_s2t[pos]

            relu = isinstance(act, nn.ReLU6)
            if isinstance(conv, nn.Identity):
                cur_channel = weights[1].size(0)
                conv, bn = identity_conv_bn(cur_channel)
            lyrs = (conv, bn)

            is_merged = pos > 0 and not pos in bumps

            if pos in self.skip_s2t and not pos in skip_bumps_s2t:
                loc_end = self.skip_s2t[pos]
                saved = (stack, weights, cparam, last, is_merged)
                weights, cparam = (weights[0], None, None, None), ("dw", 1)
                stack, last, is_merged = [], None, False

            if pos + 1 == loc_end:
                _, weights, cparam = merge_or_new(
                    stack, pos, weights, lyrs, is_merged, cparam, last
                )
                fuse_skip(weights[1])
                push_merged_layers(stack, pos + 1, weights, cparam, relu=relu, act_type="relu6")

                # Doesn't have bn (already fused), thus give `None` in the place of `bn`
                lyrs = stack[0][1], None
                relu = len(stack) > 1
                stack, weights, cparam, last, is_merged = saved

            _, weights, cparam = merge_or_new(
                stack, pos, weights, lyrs, is_merged, cparam, last
            )
            push_merged_layers(stack, pos + 1, weights, cparam, relu=relu, act_type="relu6")

            if pos + 1 == bump_end:
                relu_layer = partial(nn.ReLU6, inplace=True)
                if relu:
                    skipfeed = SkipFeed(OrderedDict(stack[:-1]), relu_layer)
                else:
                    skipfeed = SkipFeed(OrderedDict(stack))
                self.m_layers += [skipfeed]
                stack = []

            pos += 1
            last = 2 if relu else 1

        if len(stack) > 0:
            self.m_layers += [NaiveFeed(OrderedDict(stack))]
        self.m_features = nn.Sequential(*self.m_layers)

        self.to("cuda")

        result = nn.Sequential()
        for ind, blk in enumerate(self.m_features):
            result.add_module(f"blk{ind}", blk)

        pre_last = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), self.classifier]
        for ind, blk in enumerate(pre_last):
            result.add_module(f"last{ind}", blk)

        return result


def make_depth_layer_mobilenet_v2(act_ind, conv_ind, **kwargs):
    model: DepthLayerMobileNetV2 = MobileNetV2(**kwargs)

    model.__class__ = DepthLayerMobileNetV2

    model.morph()
    model.trace()

    model.fix_act(act_ind)
    model.fix_conv(conv_ind)
    model.adjust_padding(act_ind)
    return model


if __name__ == "__main__":
    model: DepthLayerMobileNetV2 = MobileNetV2()
    model.__class__ = DepthLayerMobileNetV2

    model.morph()
    model.trace()

    act_indices = {0, 50, 51}
    conv_ird_indices = set()
    for ind, (name, layer) in model.convs.items():
        if layer.in_channels != layer.out_channels or layer.stride[0] > 1:
            conv_ird_indices = set.union(conv_ird_indices, {ind})

    conv_indices = [0] + [2 * i for i in range(1, 17)] + list(conv_ird_indices)
    model.fix_act(act_indices)
    model.fix_conv(conv_indices)
    model.adjust_padding(act_indices)

    with torch.no_grad():
        x = torch.rand(1, 3, 224, 224).to("cpu")
        model.to("cpu")
        model.eval()
        print(model)
        out1 = model(x)
        merged_model = model.merge()
        merged_model.to("cpu")
        merged_model.eval()
        print(merged_model)
        out2 = merged_model(x)
        print(out1[0, :10])
        print(out2[0, :10])


