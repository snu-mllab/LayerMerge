import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_merge.models.ddpm import DDPMModel, ResnetBlock
from layer_merge.models.ddpm_datasets import dataset2config

from functools import reduce


class Swish(nn.Module):
    def nonlinearity(self, x):
        # swish
        return x * torch.sigmoid(x)

    def forward(self, x):
        return self.nonlinearity(x)


class LayerResnetBlock(ResnetBlock):
    def morph(self):
        self.nonlinear = Swish()
        self.nonlinear1 = Swish()
        self.nonlinear2 = Swish()
        self.postnorm = nn.Identity()

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

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LayerDDPMModel(DDPMModel):
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
                module.__class__ = LayerResnetBlock
                module.morph()
            # compressing from diff-pruned model
            elif module.__class__.__name__ == "ResnetBlock":
                module.__class__ = LayerResnetBlock
                module.morph()

    def trace(self):
        input_sizes, convs, res_blocks = self.get_input_and_lyrs()
        self.convs = convs
        self.res_blocks = res_blocks
        for ind, name, size in input_sizes:
            self.outs[ind - 1] = tuple(size)

        self.outs = dict(sorted(self.outs.items()))
        self.depth = len(self.convs)

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
        self.to("cpu")
        self.conv_pos = set(conv_pos)
        self.conv_pos = set.union(self.conv_pos, {self.depth})
        assert self.conv_pos.issubset(set(range(self.depth + 1)))

        new_convs = dict()
        for ind, (conv_name, _) in self.convs.items():
            if not ind in self.conv_pos:
                self.set_name(conv_name, nn.Identity())
            new_convs[ind] = (conv_name, self.get_name(conv_name))

        self.convs = new_convs
        self.to("cuda")


def make_layer_ddpm(conv_ind, dataset, **kwargs):
    model: LayerDDPMModel = DDPMModel(dataset2config(dataset))

    model.__class__ = LayerDDPMModel

    model.morph()
    model.trace()

    model.fix_conv(conv_ind)
    return model


def make_layer_ddpm_from_model(model, conv_ind, **kwargs):
    model: LayerDDPMModel = model

    model.__class__ = LayerDDPMModel

    model.morph()
    model.trace()

    model.fix_conv(conv_ind)
    return model


if __name__ == "__main__":
    dataset = "cifar10"
    model: LayerDDPMModel = DDPMModel(dataset2config(dataset))
    model.__class__ = LayerDDPMModel

    model.morph()
    model.trace()

    act_indices = set()
    conv_ird_indices = set()
    for ind, (name, layer) in model.convs.items():
        print(ind, name, layer)
        if layer.in_channels != layer.out_channels or layer.stride[0] > 1:
            conv_ird_indices = set.union(conv_ird_indices, {ind})
    
    example_indices = set(range(0, model.depth, 2))
    model.fix_conv(set.union(conv_ird_indices, example_indices))
    print(model)
    print(model.outs)
