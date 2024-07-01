from layer_merge.models.resnet import (
    ResNet,
    ResNet34,
    ResNet50,
    BasicBlock,
    Bottleneck,
)
from layer_merge.models.merge_op import NaiveFeed, SkipFeed, SkipFeedDown
from layer_merge.models.merge_op import (
    fuse_skip,
    push_merged_layers,
    unroll_conv_params,
    adjust_with_bn,
    unroll_lyrs,
)

from functools import reduce, partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerBasicBlock(BasicBlock):
    def morph(self):
        delattr(self, "relu")
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
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

        if not all(
            isinstance(layer, nn.Identity) for layer in [self.conv1, self.conv2]
        ):
            out = out + identity
        out = self.relu2(out)

        return out


class LayerBottleneck(Bottleneck):
    def morph(self):
        delattr(self, "relu")
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)
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
            diff = 0
            if isinstance(self.conv1, nn.Conv2d):
                diff += self.conv1.padding[0]
            if isinstance(self.conv2, nn.Conv2d):
                diff += self.conv2.padding[0] - 1
            x = F.pad(x, [diff] * 4)
            identity = self.downsample(x)

        # In fine-tuning stage, the size might not match
        diff = (out.shape[-1] - identity.shape[-1]) // 2
        identity = F.pad(identity, [diff] * 4)

        if not all(
            isinstance(layer, nn.Identity)
            for layer in [self.conv1, self.conv2, self.conv3]
        ):
            out = out + identity
        out = self.relu3(out)

        return out


class LayerResNet(ResNet):
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
                module.__class__ = LayerBasicBlock
                module.morph()
            elif isinstance(module, Bottleneck):
                module.__class__ = LayerBottleneck
                module.morph()
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

    def fix_conv(self, conv_pos: set):
        self.conv_pos = set(conv_pos)
        assert self.conv_pos.issubset(set(range(self.depth)))

        new_convs, new_bns, new_acts = dict(), dict(), dict()
        for ind, (conv_name, _) in self.convs.items():
            if not ind in self.conv_pos:
                self.set_name(conv_name, nn.Identity())
            new_convs[ind] = (conv_name, self.get_name(conv_name))
        for ind, (bn_name, _) in self.bns.items():
            if not ind in self.conv_pos:
                self.set_name(bn_name, nn.Identity())
            new_bns[ind] = (bn_name, self.get_name(bn_name))
        for ind, (act_name, _) in self.acts.items():
            if not ind + 1 in self.conv_pos and ind + 1 in self.skip_s2t.values():
                self.set_name(act_name, nn.Identity())
            if not ind in self.conv_pos and not ind in self.skip_s2t.values():
                self.set_name(act_name, nn.Identity())
            new_acts[ind] = (act_name, self.get_name(act_name))

        self.convs = new_convs
        self.bns = new_bns
        self.acts = new_acts

    def get_skip_bumps(self):
        skip_bumps, skip_bumps_s2t = set(), dict()
        for src, tgt in self.skip_s2t.items():
            # Compute the set {x: src<x<=tgt}
            range_set = set(range(src + 1, tgt + 1))

            # Compute the intersection with stop_pos
            intersection = range_set.intersection(self.conv_pos)

            # If the size of intersection is greater or equal to 2, add it to cleaned_skip_s2t
            if len(intersection) >= 2:
                skip_bumps_s2t[src] = tgt
                skip_bumps.update([src, tgt])
        return skip_bumps, skip_bumps_s2t

    def merge(self):
        self.to("cpu")
        self.is_merged = True

        _, skip_bumps_s2t = self.get_skip_bumps()

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

            if pos in self.skip_s2t and not pos in skip_bumps_s2t:
                loc_end = self.skip_s2t[pos]
                saved = (stack, weights, cparam, last)
                weights, cparam = (weights[0], None, None, None), ("dw", 1)
                stack, last = [], None
            if pos + 1 == loc_end:
                if not isinstance(conv, nn.Identity):
                    _, weights, cparam = unroll_lyrs(pos, weights, lyrs)
                else:
                    stack = []
                if not None in weights:
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
                stack, weights, cparam, last = saved

            if not isinstance(lyrs[0], nn.Identity):
                _, weights, cparam = unroll_lyrs(pos, weights, lyrs)
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


def make_layer_resnet(arch, class_num, enable_bias, conv_ind):
    if arch == "resnet34":
        model: LayerResNet = ResNet34(class_num, enable_bias)
    elif arch == "resnet50":
        model: LayerResNet = ResNet50(class_num, enable_bias)
    else:
        raise NotImplementedError()

    model.__class__ = LayerResNet

    model.morph()
    model.trace()

    model.fix_conv(conv_ind)
    return model


if __name__ == "__main__":
    model: LayerResNet = ResNet34()
    model.__class__ = LayerResNet

    model.morph()
    model.trace()

    act_indices = set()
    conv_ird_indices = set()
    for ind, (name, layer) in model.convs.items():
        if layer.in_channels != layer.out_channels or layer.stride[0] > 1:
            conv_ird_indices = set.union(conv_ird_indices, {ind})

    conv_indices = [0] + [3 * i + 1 for i in range(1, 10)] + list(conv_ird_indices)
    model.fix_conv(conv_indices)
    with open("model.txt", "w") as f:
        f.write(str(model))
    merged_model = model.merge()
    with open("merged_model.txt", "w") as f:
        f.write(str(merged_model))

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
