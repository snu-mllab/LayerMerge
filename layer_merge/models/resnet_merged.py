from layer_merge.models.resnet import ResNet, ResNet34, ResNet50, BasicBlock, Bottleneck
from layer_merge.models.merge_op import NaiveFeed, SkipFeed, SkipFeedDown
from layer_merge.models.merge_op import (
    merge_or_new,
    fuse_skip,
    push_merged_layers,
    unroll_conv_params,
    adjust_with_bn,
)

from functools import reduce, partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthBasicBlock(BasicBlock):
    def morph(self):
        delattr(self, "relu")
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        order = ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2", "downsample"]
        new_modules = []
        for key in order:
            if key in self._modules:
                new_modules.append((key, self._modules[key]))
        self._modules = OrderedDict(new_modules)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # In fine-tuning stage, the size might not match
        diff = (out.shape[-1] - identity.shape[-1]) // 2
        identity = F.pad(identity, [diff] * 4)

        out = out + identity
        out = self.relu2(out)

        return out


class DepthBottleneck(Bottleneck):
    def morph(self):
        delattr(self, "relu")
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        order = [
            "conv1",
            "bn1",
            "relu1",
            "conv2",
            "bn2",
            "relu2",
            "conv3",
            "bn3",
            "relu3",
            "downsample",
        ]
        new_modules = []
        for key in order:
            if key in self._modules:
                new_modules.append((key, self._modules[key]))
        self._modules = OrderedDict(new_modules)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            # In fine-tuning stage, the size might not match
            diff = self.conv1.padding[0] + self.conv2.padding[0] - 1
            x = F.pad(x, [diff] * 4)
            identity = self.downsample(x)

        # In fine-tuning stage, the size might not match
        diff = (out.shape[-1] - identity.shape[-1]) // 2
        identity = F.pad(identity, [diff] * 4)

        out = out + identity
        out = self.relu3(out)

        return out


class DepthResNet(ResNet):
    def morph(self):
        # layers
        self.cur_index = -1
        self.convs, self.bns, self.acts = {}, {}, {}
        self.downs, self.down_bns = {}, {}
        self.outs = {}
        # network config
        self.skip_s2t = {}
        self.iden_pos = set()
        self.depth = -1
        for module in self.modules():
            if isinstance(module, BasicBlock):
                module.__class__ = DepthBasicBlock
                module.morph()
            elif isinstance(module, Bottleneck):
                module.__class__ = DepthBottleneck
                module.morph()
        self.act_pos = set()
        self.str_pos = set()

    def trace(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if "downsample" in name:
                    self.downs[self.cur_index] = (name, module)
                else:
                    self.cur_index += 1
                    self.convs[self.cur_index] = (name, module)
            if isinstance(module, nn.BatchNorm2d):
                if "downsample" in name:
                    self.down_bns[self.cur_index] = (name, module)
                else:
                    self.bns[self.cur_index] = (name, module)
            if isinstance(module, nn.ReLU):
                self.acts[self.cur_index] = (name, module)

        self.depth = len(self.acts)

        if self.depth + 1 == 34:
            self.skip_s2t = {i: i + 2 for i in range(0, 32, 2)}
            self.str_pos = {0, 7, 15, 27}
        elif self.depth + 1 == 50:
            self.skip_s2t = {i: i + 3 for i in range(0, 48, 3)}
            self.str_pos = {0, 11, 23, 41}
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
        self.act_pos = set(act_pos)
        assert self.act_pos.issubset(set(range(self.depth)))
        new_acts = dict()
        for ind, (act_name, _) in self.acts.items():
            if ind in self.act_pos:
                self.set_name(act_name, nn.ReLU(inplace=True))
            else:
                self.set_name(act_name, nn.Identity())
            new_acts[ind] = (act_name, self.get_name(act_name))
        self.acts = new_acts

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

        stop_bumps = set.union(self.merge_pos, self.str_pos)
        skip_bumps, _ = self.get_skip_bumps(stop_bumps)
        bumps = set.union(skip_bumps, stop_bumps)
        for pos, (_, layer) in self.convs.items():
            if pos == 0:
                continue
            node_pos = pos - 1
            if starting_layer == None:
                starting_layer = layer
            if node_pos in bumps and not node_pos == 0:
                starting_layer.padding = (pad, pad)
                pad = 0
                starting_layer = layer
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

            relu = isinstance(act, nn.ReLU)
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
                if pos + 1 in self.downs:
                    down = self.downs[pos + 1][1]
                    down_bn = self.down_bns[pos + 1][1]

                    cw, bw = unroll_conv_params(down)
                    m_b = torch.zeros(cw.size(0))
                    if bw != None:
                        m_b += bw
                    m_cw, m_b = adjust_with_bn(cw, m_b, down_bn)

                    weights_list = list(weights)
                    mid = weights_list[1].size(2) // 2
                    weights_list[1][:, :, mid, mid] += m_cw[:, :, 0, 0]
                    weights_list[3] += m_b
                    weights = tuple(weights_list)
                else:
                    fuse_skip(weights[1])
                push_merged_layers(stack, pos + 1, weights, cparam, relu=relu)

                # Doesn't have bn (already fused), thus give `None` in the place of `bn`
                lyrs = stack[0][1], None
                relu = len(stack) > 1
                stack, weights, cparam, last, is_merged = saved

            _, weights, cparam = merge_or_new(
                stack, pos, weights, lyrs, is_merged, cparam, last
            )
            push_merged_layers(stack, pos + 1, weights, cparam, relu=relu)
            if pos == -1:
                stack.append(
                    ("maxpool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                )

            if pos + 1 == bump_end:
                relu_layer = partial(nn.ReLU, inplace=True)
                if pos + 1 in self.downs:
                    down = self.downs[pos + 1][1]
                    down_bn = self.down_bns[pos + 1][1]

                    cw, bw = unroll_conv_params(down)
                    m_b = torch.zeros(cw.size(0))
                    if bw != None:
                        m_b += bw
                    m_cw, m_b = adjust_with_bn(cw, m_b, down_bn)
                    new_down = nn.Conv2d(
                        m_cw.size(1),
                        m_cw.size(0),
                        kernel_size=m_cw.size(2),
                        stride=down.stride,
                        padding=down.padding,
                        bias=True,
                        groups=down.groups,
                    )
                    new_down.weight.data = m_cw.clone().detach()
                    new_down.bias.data = m_b.clone().detach()

                    if relu:
                        skipfeed = SkipFeedDown(
                            OrderedDict(stack[:-1]), relu_layer, new_down
                        )
                    else:
                        skipfeed = SkipFeedDown(
                            OrderedDict(stack), nn.Identity, new_down
                        )
                elif relu and not (pos + 1 in self.downs):
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

        pre_last = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), self.fc]
        for ind, blk in enumerate(pre_last):
            result.add_module(f"last{ind}", blk)

        return result


def make_depth_resnet(arch, class_num, enable_bias, act_pos, merge_pos):
    if arch == "resnet34":
        model: DepthResNet = ResNet34(class_num, enable_bias)
    elif arch == "resnet50":
        model: DepthResNet = ResNet50(class_num, enable_bias)
    else:
        raise NotImplementedError()

    model.__class__ = DepthResNet

    model.morph()
    model.trace()

    model.fix_act(act_pos)
    model.adjust_padding(merge_pos)
    return model


if __name__ == "__main__":
    model: DepthResNet = ResNet50()
    model.__class__ = DepthResNet

    model.morph()
    model.trace()

    # act_indices = set([2 * i for i in range(17)])
    act_indices = set()
    model.fix_act(act_indices)
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
