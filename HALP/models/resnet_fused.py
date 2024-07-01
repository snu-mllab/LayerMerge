from models.resnet import BasicBlock, Bottleneck, ResNet
from models.resnet_pruned import (
    BasicBlock as PrunedBasicBlock,
    Bottleneck as PrunedBottleneck,
)
from layer_merge.models.merge_op import (
    unroll_conv_params,
    adjust_with_bn,
)
from collections import OrderedDict

import torch
import torch.nn as nn


class FusedBasicBlock(BasicBlock):
    def fuse_conv_bn(self, conv, bn, name):
        cw, bw = unroll_conv_params(conv)
        m_b = torch.zeros(cw.size(0))
        if bw != None:
            m_b += bw
        m_cw, m_b = adjust_with_bn(cw, m_b, bn)

        new_conv = nn.Conv2d(
            m_cw.size(1),
            m_cw.size(0),
            kernel_size=m_cw.size(2),
            stride=conv.stride,
            padding=conv.padding,
            bias=True,
            groups=conv.groups,
        )
        new_conv.weight.data = m_cw.clone().detach()
        new_conv.bias.data = m_b.clone().detach()
        setattr(self, name, new_conv)

    def fuse(self):
        if self.downsample != None:
            conv, bn = self.downsample[0], self.downsample[1]
            delattr(self, "downsample")
            self.fuse_conv_bn(conv, bn, "downsample")

        conv, bn = self.conv1, self.bn1
        delattr(self, "conv1")
        delattr(self, "bn1")
        self.fuse_conv_bn(conv, bn, "conv1")

        conv, bn = self.conv2, self.bn2
        delattr(self, "conv2")
        delattr(self, "bn2")
        self.fuse_conv_bn(conv, bn, "conv2")

        order = ["conv1", "conv2", "relu", "downsample"]
        new_modules = []
        for key in order:
            if key in self._modules:
                new_modules.append((key, self._modules[key]))
        self._modules = OrderedDict(new_modules)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FusedBottleneck(Bottleneck):
    def fuse_conv_bn(self, conv, bn, name):
        cw, bw = unroll_conv_params(conv)
        m_b = torch.zeros(cw.size(0))
        if bw != None:
            m_b += bw
        m_cw, m_b = adjust_with_bn(cw, m_b, bn)

        new_conv = nn.Conv2d(
            m_cw.size(1),
            m_cw.size(0),
            kernel_size=m_cw.size(2),
            stride=conv.stride,
            padding=conv.padding,
            bias=True,
            groups=conv.groups,
        )
        new_conv.weight.data = m_cw.clone().detach()
        new_conv.bias.data = m_b.clone().detach()
        setattr(self, name, new_conv)

    def fuse(self):
        if self.downsample != None:
            conv, bn = self.downsample[0], self.downsample[1]
            delattr(self, "downsample")
            self.fuse_conv_bn(conv, bn, "downsample")

        conv, bn = self.conv1, self.bn1
        delattr(self, "conv1")
        delattr(self, "bn1")
        self.fuse_conv_bn(conv, bn, "conv1")

        conv, bn = self.conv2, self.bn2
        delattr(self, "conv2")
        delattr(self, "bn2")
        self.fuse_conv_bn(conv, bn, "conv2")

        conv, bn = self.conv3, self.bn3
        delattr(self, "conv3")
        delattr(self, "bn3")
        self.fuse_conv_bn(conv, bn, "conv3")

        order = ["conv1", "conv2", "conv3", "relu", "downsample"]
        new_modules = []
        for key in order:
            if key in self._modules:
                new_modules.append((key, self._modules[key]))
        self._modules = OrderedDict(new_modules)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FusedEmptyBlock(nn.Module):
    def fuse(self):
        layers = ["conv1", "bn1", "conv2", "bn2", "relu2", "conv3", "bn3"]
        for name in layers:
            if hasattr(self, name):
                delattr(self, name)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.add_bias + identity
        out = self.relu(out)
        return out


class FusedResNet(ResNet):
    def fuse_conv_bn(self, conv, bn, name):
        cw, bw = unroll_conv_params(conv)
        m_b = torch.zeros(cw.size(0))
        if bw != None:
            m_b += bw
        m_cw, m_b = adjust_with_bn(cw, m_b, bn)

        new_conv = nn.Conv2d(
            m_cw.size(1),
            m_cw.size(0),
            kernel_size=m_cw.size(2),
            stride=conv.stride,
            padding=conv.padding,
            bias=True,
            groups=conv.groups,
        )
        new_conv.weight.data = m_cw.clone().detach()
        new_conv.bias.data = m_b.clone().detach()
        setattr(self, name, new_conv)

    def fuse(self, block=None):
        conv, bn = self.conv1, self.bn1
        delattr(self, "conv1")
        delattr(self, "bn1")
        self.fuse_conv_bn(conv, bn, "conv1")
        if hasattr(self, "relu2"):
            delattr(self, "relu2")

        for module in self.modules():
            if isinstance(module, (BasicBlock, PrunedBasicBlock)):
                if module.conv1 is not None:
                    module.__class__ = FusedBasicBlock
                else:
                    # Support layer pruning in HALP
                    module.__class__ = FusedEmptyBlock
                module.fuse()
            elif isinstance(module, (Bottleneck, PrunedBottleneck)):
                if module.conv1 is not None and module.conv2 is not None:
                    module.__class__ = FusedBottleneck
                else:
                    # Support layer pruning in HALP
                    module.__class__ = FusedEmptyBlock
                module.fuse()

        order = [
            "conv1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
            "fc",
        ]
        new_modules = []
        for key in order:
            if key in self._modules:
                new_modules.append((key, self._modules[key]))
        self._modules = OrderedDict(new_modules)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def fuse_resnet(model: FusedResNet):
    model.to("cpu")
    model.__class__ = FusedResNet
    model.fuse()
    model.eval()
    model.to("cuda")
    return model


if __name__ == "__main__":
    from models.resnet import ResNet34, ResNet50

    model: FusedResNet = ResNet50()
    model.__class__ = FusedResNet

    with torch.no_grad():
        x = torch.rand(1, 3, 224, 224).to("cpu")
        model.to("cpu")
        model.eval()
        print(model)
        out1 = model(x)

        model.fuse()
        model.eval()

        print(model)
        out2 = model(x)
        print(out1[0, :10])
        print(out2[0, :10])
