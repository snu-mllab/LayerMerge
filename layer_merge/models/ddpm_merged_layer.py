import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_merge.models.ddpm import DDPMModel, ResnetBlock
from layer_merge.models.ddpm_datasets import dataset2config
from layer_merge.models.merge_op import (
    unroll_conv_params,
    update_m_weights,
    identity_conv_bn,
)

from functools import reduce


class Padding(nn.Identity):
    def __init__(self, padding, out_channels):
        super(Padding, self).__init__()
        self.padding = padding
        self.out_channels = out_channels

    def forward(self, x):
        pad = (self.padding[0], self.padding[1], self.padding[0], self.padding[1])
        return F.pad(x, pad)
    
    def extra_repr(self):
        return "padding={}, out_channels={}".format(self.padding, self.out_channels)


class Swish(nn.Module):
    def nonlinearity(self, x):
        # swish
        return x * torch.sigmoid(x)

    def forward(self, x):
        return self.nonlinearity(x)


class DepthLayerResnetBlock(ResnetBlock):
    def morph(self):
        self.nonlinear = Swish()
        self.nonlinear1 = Swish()
        self.nonlinear2 = Swish()
        self.postnorm = nn.Identity()

    def fix_act(self, act_pos: set):
        assert act_pos.issubset(set(range(2)))
        self.act_pos = act_pos
        self.postnorm = nn.Identity()
        if not 1 in act_pos:
            self.nonlinear2 = nn.Identity()
            self.norm2 = nn.Identity()

            if not isinstance(self.conv2, nn.Conv2d):
                out_channels = self.conv1.out_channels
            else:
                out_channels = self.conv2.out_channels
            
            if out_channels % 32 == 0:
                self.postnorm = nn.GroupNorm(32, out_channels)
            else:
                self.postnorm = nn.GroupNorm(out_channels, out_channels)

            conv1_pad = self.conv1.padding[0]
            conv2_pad = self.conv2.padding[0]
            merged_conv_pad = conv1_pad + conv2_pad
            self.conv1.padding = (merged_conv_pad, merged_conv_pad)
            self.conv2.padding = (0, 0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.nonlinear1(h)
        h = self.conv1(h)

        h = h + self.temb_proj(self.nonlinear(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.nonlinear2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.postnorm(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class DepthLayerResnetBlockMerged(DepthLayerResnetBlock):
    def merge(self):
        self.to("cpu")
        if isinstance(self.conv1, nn.Identity):
            padding = self.conv1.padding
            out_channels = self.conv1.out_channels
            self.conv1, _ = identity_conv_bn(out_channels)
            self.conv1.bias = nn.Parameter(torch.zeros(out_channels))
            self.conv1.padding = padding
        if isinstance(self.conv2, nn.Identity):
            padding = self.conv2.padding
            out_channels = self.conv2.out_channels
            self.conv2, _ = identity_conv_bn(out_channels, dw=False)
            self.conv2.bias = nn.Parameter(torch.zeros(out_channels))
            self.conv2.padding = padding

        weight = self.temb_proj.weight
        bias = self.temb_proj.bias

        weight = torch.einsum("jikl,im->jm", self.conv2.weight, weight)
        bias = torch.einsum("jikl,i->j", self.conv2.weight, bias)

        cw, bw = unroll_conv_params(
            self.conv1, dw=self.conv1.in_channels == self.conv1.groups
        )

        _, m_cw, m_p, m_bw = update_m_weights(
            isize=0,
            m_cw=cw.clone().detach(),
            m_p=self.conv1.padding[0],
            m_b=bw.clone().detach(),
            conv=self.conv2,
            bn=None,
            merged=True,
        )

        self.merged_conv = nn.Conv2d(
            self.conv1.in_channels,
            self.conv2.out_channels,
            kernel_size=m_cw.size(2),
            stride=1,
            padding=m_p,
            bias=True,
        )
        self.merged_conv.weight = nn.Parameter(m_cw)
        self.merged_conv.bias = nn.Parameter(m_bw)
        rm_modules = ["conv1", "conv2", "norm2", "dropout", "nonlinear2"]
        for rm_md in rm_modules:
            delattr(self, rm_md)

        # Update temb_proj weights and bias
        self.temb_proj.weight = nn.Parameter(weight)
        self.temb_proj.bias = nn.Parameter(bias)
        self.to("cuda")

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.nonlinear1(h)

        h = self.merged_conv(h)
        # reweighted embedding projection
        h = h + self.temb_proj(self.nonlinear(temb))[:, :, None, None]
        h = self.postnorm(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class DepthLayerDDPMModel(DDPMModel):
    def get_input_and_lyrs(self):
        dummy = torch.randn(128, 3, self.resolution, self.resolution)
        dummy_t = torch.rand(128)
        input_sizes = []
        cur_index = [0]
        convs = dict()
        res_blocks = dict()

        def _conv_hook(module, input, output):
            # This function will be called during the forward pass
            cur_index[0] += 1
            input_sizes.append((cur_index[0], module.name, input[0].size()))
            convs[cur_index[0]] = (module.name, module)

        def _res_hook(module, input, output):
            res_blocks[cur_index[0] - 1] = (module.name, module)

        # Register the hook
        hooks = []
        for name, module in self.named_modules():
            if (
                isinstance(module, nn.Conv2d)
                and "conv" in name
                and not "shortcut" in name
            ):
                module.name = name
                hooks.append(module.register_forward_hook(_conv_hook))
            elif isinstance(module, ResnetBlock):
                module.name = name
                hooks.append(module.register_forward_hook(_res_hook))

        # Forward pass
        self.to("cuda")
        dummy, dummy_t = dummy.to("cuda"), dummy_t.to("cuda")
        with torch.no_grad():
            self.forward(dummy, dummy_t)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        for name, module in self.named_modules():
            if (
                isinstance(module, nn.Conv2d)
                and "conv" in name
                and not "shortcut" in name
            ) or isinstance(module, ResnetBlock):
                delattr(module, "name")

        # Return the collected input sizes and convs
        return input_sizes, convs, res_blocks

    def morph(self):
        # layers
        self.convs = {}
        self.outs = {}
        self.iden_pos = set()

        for module in self.modules():
            if isinstance(module, ResnetBlock):
                module.__class__ = DepthLayerResnetBlock
                module.morph()
            # compressing from diff-pruned model
            elif module.__class__.__name__ == "ResnetBlock":
                module.__class__ = DepthLayerResnetBlock
                module.morph()

    def trace(self):
        input_sizes, convs, res_blocks = self.get_input_and_lyrs()
        self.convs = convs
        self.res_blocks = res_blocks
        for ind, name, size in input_sizes:
            self.outs[ind - 1] = tuple(size)

        self.outs = dict(sorted(self.outs.items()))
        self.depth = len(self.convs)
        self.skip_s2t = {
            1: 49,
            3: 47,
            5: 45,
            6: 42,
            8: 40,
            10: 38,
            11: 35,
            13: 33,
            15: 31,
            16: 28,
            18: 26,
            20: 24,
            1: 3,
            3: 5,
            6: 8,
            8: 10,
            11: 13,
            13: 15,
            16: 18,
            18: 20,
            20: 22,
            22: 24,
            24: 26,
            26: 28,
            28: 30,
            31: 33,
            33: 35,
            35: 37,
            38: 40,
            40: 42,
            42: 44,
            45: 47,
            47: 49,
        }
        self.attn_pos = {8, 10, 22, 40, 42, 44}
        self.upsample_pos = {31, 38, 45}

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
        self.to("cpu")
        self.act_pos = set(act_pos)
        skip_src, skip_tgt = set(self.skip_s2t.keys()), set(self.skip_s2t.values())
        self.bumps = set.union(skip_src, skip_tgt, self.attn_pos)
        self.act_pos = set.union(self.act_pos, self.bumps, {0}, {self.depth})

        for ind, (_, res_blk) in self.res_blocks.items():
            if ind in self.act_pos:
                res_blk.fix_act({0, 1})
            else:
                res_blk.fix_act({0})
        self.to("cuda")

    def fix_conv(self, conv_pos: set):
        self.to("cpu")
        self.conv_pos = set(conv_pos)
        self.conv_pos = set.union(self.conv_pos, {self.depth})
        assert self.conv_pos.issubset(set(range(self.depth + 1)))

        new_convs = dict()
        for ind, (conv_name, conv_layer) in self.convs.items():
            if not ind in self.conv_pos:
                self.set_name(
                    conv_name, Padding(padding=(0, 0), out_channels=conv_layer.out_channels)
                )
            new_convs[ind] = (conv_name, self.get_name(conv_name))

        self.convs = new_convs
        self.to("cuda")

    def merge(self):
        self.to("cpu")
        with torch.no_grad():
            for ind, (_, res_blk) in self.res_blocks.items():
                if not ind in self.act_pos:
                    res_blk: DepthLayerResnetBlockMerged = res_blk
                    res_blk.__class__ = DepthLayerResnetBlockMerged
                    res_blk.merge()
        self.to("cuda")


def make_depth_layer_ddpm(act_ind, conv_ind, dataset, **kwargs):
    model: DepthLayerDDPMModel = DDPMModel(dataset2config(dataset))

    model.__class__ = DepthLayerDDPMModel

    model.morph()
    model.trace()

    model.fix_conv(conv_ind)
    model.fix_act(act_ind)
    return model


def make_depth_layer_ddpm_from_model(model, act_ind, conv_ind, **kwargs):
    model: DepthLayerDDPMModel = model

    model.__class__ = DepthLayerDDPMModel

    model.morph()
    model.trace()

    model.fix_conv(conv_ind)
    model.fix_act(act_ind)
    return model


if __name__ == "__main__":
    dataset = "cifar10"

    inp = torch.rand(1, 3, 32, 32)
    temb = torch.rand(1)

    act_indices = set([0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 22, 24, 26, 27, 28, 30, 31, 33, 35, 37, 38, 40, 42, 44, 45, 47, 49, 51])
    conv_indices = set([0, 1, 3, 4, 6, 7, 9, 11, 13, 14, 16, 17, 19, 21, 24, 25, 27, 28, 29, 31, 32, 34, 36, 38, 39, 41, 43, 45, 46, 48, 50])
    print(act_indices)
    print(conv_indices)


    model = make_depth_layer_ddpm(act_indices, conv_indices, dataset)

    model.eval()

    with torch.no_grad():
        model.to("cpu")
        model.eval()
        with open("ddpm_chk.txt", "w") as f:
            f.write(str(model))
        oup1 = model(inp, temb)

        model.merge()
        model.to("cpu")
        model.eval()
        with open("ddpm_mgd.txt", "w") as f:
            f.write(str(model))
        oup2 = model(inp, temb)

    print(oup1[0, 0, :3, :3])
    print(oup2[0, 0, :3, :3])
    print(torch.norm(oup1 - oup2))
