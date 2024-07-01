from torch import nn
from einops import rearrange, repeat
from typing import List
from functools import reduce
from colorama import Fore, Style
from torchvision.ops.misc import Conv2dNormActivation
from collections import OrderedDict

import warnings
import torch
import torch.nn.functional as F
import re


def unroll_bn_params(bn):
    bw = bn.weight.data.clone().detach()
    bb = bn.bias.data.clone().detach()
    brm = bn.running_mean.data.clone().detach()
    brv = bn.running_var.data.clone().detach()
    return bw, bb, brm, brv


def unroll_conv_params(conv, dw=False):
    # params of conv (if dw then to dense)
    if dw:
        cw = conv.weight.data.clone().detach()
        cw = repeat(cw, "c one h w -> c (rep one) h w", rep=cw.shape[0])
        cw = torch.diagonal(cw, dim1=0, dim2=1)
        cw = torch.diag_embed(cw, dim1=0, dim2=1)
    else:
        cw = conv.weight.data.clone().detach()

    if conv.bias == None:
        bw = None
    else:
        bw = conv.bias.data.clone().detach()

    return cw, bw


def unroll_bias_params(bias):
    biasw = bias.weight.data.clone().detach()
    return biasw


def adjust_with_bn(conv_weight, bias, bn):
    # Address bn
    bw, bb, brm, brv = unroll_bn_params(bn)
    for i in range(bb.size(0)):
        bias[i] *= bw[i] / torch.sqrt(brv[i] + 1e-5)
        conv_weight[i] = conv_weight[i] / torch.sqrt(brv[i] + 1e-5) * bw[i]
        bb[i] -= brm[i] / torch.sqrt(brv[i] + 1e-5) * bw[i]
    for i in range(bb.size(0)):
        bias[i] += bb[i]
    return conv_weight, bias


def conv_with_new(old_conv, old_bias, new_conv, stride, padding):
    old_conv = rearrange(old_conv, "oup inp h w -> inp oup h w")
    f_new_conv = torch.flip(new_conv, [2, 3])
    old_conv = F.conv2d(
        input=old_conv, weight=f_new_conv, stride=1, padding=f_new_conv.shape[2] - 1
    )
    old_conv = rearrange(old_conv, "inp oup h w -> oup inp h w")
    old_bias = torch.einsum("jikl,i->j", new_conv, old_bias)
    return old_conv, old_bias


# `stack` is called by reference; pop/push reflects outside the function
def push_layer(stack, new_name, weight, mode, stride=1, padding=0, bias=None):
    if mode == "cw":
        layer = nn.Conv2d(
            weight.size(1),
            weight.size(0),
            kernel_size=weight.size(2),
            stride=stride,
            padding=padding,
            bias=(bias != None),
        )
        layer.weight.data = weight.clone().detach()
        if bias != None:
            layer.bias.data = bias.clone().detach()
    elif mode == "dw":
        assert weight.size(1) == weight.size(0)
        layer = nn.Conv2d(
            weight.size(1),
            weight.size(0),
            kernel_size=weight.size(2),
            stride=stride,
            padding=padding,
            bias=(bias != None),
            groups=weight.size(1),
        )
        weight = torch.diagonal(weight, dim1=0, dim2=1)
        weight = rearrange(weight, "h w (c one) -> c one h w", one=1)
        layer.weight.data = weight.clone().detach()
        if bias != None:
            layer.bias.data = bias.clone().detach()
    elif mode == "relu6":
        layer = nn.ReLU6(inplace=True)
    elif mode == "relu":
        layer = nn.ReLU(inplace=True)
    else:
        raise NotImplementedError("Not implemented mode")

    if isinstance(stack, list):
        stack.append((new_name, layer))
    elif isinstance(stack, nn.Sequential):
        stack.add_module(new_name, layer)
    else:
        raise NotImplementedError("Not implemented type of stack")


def get_mid(tensor: torch.Tensor, reduced_axes: List[int]):
    sz = list(tensor.shape)
    new_sz = sz.copy()
    result = tensor
    ind = [slice(None)] * len(sz)
    for i in reduced_axes:
        new_sz[i] = 1
        ind[i] = sz[i] // 2
        result = result[ind].view(new_sz)
        ind[i] = slice(None)
    return result


def push_merged_layers(m_layers, pos, weights, cparam, relu=False, act_type="relu6"):
    assert act_type in ["relu6", "relu"]
    _, m_cw, m_p, m_b = weights
    ctype, cstr = cparam
    push_layer(m_layers, f"merged_conv{pos}", m_cw, ctype, cstr, m_p, m_b)
    if relu:
        push_layer(m_layers, f"relu{pos}", None, act_type)


def update_m_weights(isize, m_cw, m_p, m_b, conv, bn, merged):
    cw, bw = unroll_conv_params(conv, conv.in_channels == conv.groups)
    # feature size after conv
    isize = (isize - cw.size(2) + 2 * conv.padding[0]) // conv.stride[0] + 1

    # Convolve cw (w/ flip) and add bw
    if merged:
        m_p += conv.padding[0]
        m_cw, m_b = conv_with_new(m_cw, m_b, cw, conv.stride[0], conv.padding[0])
    else:
        m_p = conv.padding[0]
        m_cw = cw
        m_b = torch.zeros(cw.size(0))

    if bw != None:
        m_b += bw

    assert isinstance(bn, nn.BatchNorm2d) or bn == None
    if isinstance(bn, nn.BatchNorm2d):
        m_cw, m_b = adjust_with_bn(m_cw, m_b, bn)

    return isize, m_cw, m_p, m_b


def merge_or_new(m_layers, pos, weights, lyrs, merged, cparam, pop_k=2):
    """
    - if `merged` == True :
        - Pop the last (conv, bn) in `m_layers` (have same params with `weights`).
        - Merge the `weights` with `lyrs` and return the merged params.
    - if `merged` == False :
        - Return the parameters of `lyrs`.
    """
    isize, m_cw, m_p, m_b = weights
    ctype, cstr = cparam
    conv, bn = lyrs

    pos += 1
    isize, m_cw, m_p, m_b = update_m_weights(isize, m_cw, m_p, m_b, conv, bn, merged)

    new_type = "dw" if conv.in_channels == conv.groups else "cw"
    new_str = conv.stride[0]

    if merged:
        # pop k layers from the back
        del m_layers[-pop_k:]
        pos -= 1
        ctype = "dw" if all(tp == "dw" for tp in [new_type, ctype]) else "cw"
        cstr = cstr * new_str
    else:
        ctype = new_type
        cstr = new_str

    return pos, (isize, m_cw, m_p, m_b), (ctype, cstr)


def fuse_skip(cw):
    # Fuse identity addition
    mid = cw.size(2) // 2
    for i in range(cw.size(1)):
        cw[i][i][mid][mid] += 1


def get_skip_info(blocks, arch="mbv2"):
    assert arch in ["mbv2"]

    node_pos = 0
    skip_s2t, skip_t2s = dict(), dict()
    str_pos = set()
    for block in blocks:
        if arch == "mbv2" and block.use_res_connect:
            src = node_pos
        for layer in block.conv.modules():
            if isinstance(layer, nn.Conv2d):
                node_pos += 1
                # merging strided conv is not implemented yet
                if layer.stride != (1, 1):
                    if arch == "mbv2":
                        str_pos.add(node_pos + 1)
        if arch == "mbv2" and block.use_res_connect:
            skip_s2t[src], skip_t2s[node_pos] = node_pos, src
    return skip_s2t, skip_t2s, str_pos


def get_skip_bumps(act_pos, skip_s2t):
    ind, acts = 0, sorted(list(act_pos))
    bumps, bumps_s2t, l = set(), dict(), len(acts)
    if l == 0:
        return bumps, bumps_s2t
    for src, tgt in skip_s2t.items():
        while acts[ind] < tgt:
            if acts[ind] > src:
                bumps.update([src, tgt])
                bumps_s2t[src] = tgt
                break
            ind += 1
            if ind >= l:
                return bumps, bumps_s2t
    return bumps, bumps_s2t


def adjust_padding(blocks, bumps):
    node_pos, pad, starting_layer = 0, 0, None
    for block in blocks:
        for layer in block.conv:
            if isinstance(layer, nn.Conv2d):
                if starting_layer == None:
                    starting_layer = layer
                if node_pos in bumps and not node_pos == 0:
                    starting_layer.padding = (pad, pad)
                    pad = 0
                    starting_layer = layer
                node_pos += 1
                pad += layer.padding[0]
                layer.padding = (0, 0)
        starting_layer.padding = (pad, pad)


def adjust_isize(blocks):
    for ind, block in enumerate(blocks):
        if ind == 0:
            cur_isize = block.isize
        else:
            block.isize = cur_isize
        for layer in block.conv:
            if isinstance(layer, nn.Conv2d):
                k, pad, st = layer.kernel_size[0], layer.padding[0], layer.stride[0]
                cur_isize = (cur_isize - k + 2 * pad) // st + 1


def add_nonlinear(blocks, add_pos):
    node_pos = 0
    for block in blocks:
        for layer in block.conv:
            if isinstance(layer, nn.Conv2d):
                node_pos += 1
        if node_pos in add_pos:
            block.conv.add_module("relu3", nn.ReLU6(inplace=True))


def fuse_bn(module):
    module.to("cpu")
    # Input module is merged module
    prev_lyr = None
    remove_bns = []
    for name, lyr in module.named_modules():
        # Fuse batchnorm layers with previous conv layers
        if "md.bn" in name:
            assert isinstance(prev_lyr, nn.Conv2d) and isinstance(lyr, nn.BatchNorm2d)
            cw, bw = unroll_conv_params(
                prev_lyr, prev_lyr.in_channels == prev_lyr.groups
            )
            m_b = torch.zeros(cw.size(0))
            if bw != None:
                m_b += bw
            m_cw, m_b = adjust_with_bn(cw, m_b, lyr)

            new_conv = nn.Conv2d(
                m_cw.size(1),
                m_cw.size(0),
                kernel_size=m_cw.size(2),
                stride=prev_lyr.stride,
                padding=prev_lyr.padding,
                bias=True,
                groups=prev_lyr.groups,
            )
            # depthwise conv
            if prev_lyr.in_channels == prev_lyr.groups:
                m_cw = torch.diagonal(m_cw, dim1=0, dim2=1)
                m_cw = rearrange(m_cw, "h w (c one) -> c one h w", one=1)

            new_conv.weight.data = m_cw.clone().detach()
            new_conv.bias.data = m_b.clone().detach()

            conv_names = prev_name.split(".")
            seq = reduce(getattr, conv_names[:-1], module)
            setattr(seq, conv_names[-1], new_conv)

            remove_bns += [name]
        prev_lyr = lyr
        prev_name = name
    module.to("cuda")

    for remove_bn in remove_bns:
        bn_names = remove_bn.split(".")
        seq = reduce(getattr, bn_names[:-1], module)
        delattr(seq, bn_names[-1])


def simulate_merge(conv1: nn.Conv2d, conv2: nn.Conv2d):
    assert all(isinstance(x, nn.Conv2d) for x in [conv1, conv2])
    new_in_channels = conv1.in_channels
    new_out_channels = conv2.out_channels
    new_kernel = (conv1.kernel_size[0] // 2 + conv2.kernel_size[0] // 2) * 2 + 1
    new_stride = conv1.stride[0] * conv2.stride[0]
    new_pad = conv1.padding[0] + conv2.padding[0]
    is_dw = all(x.groups == x.in_channels for x in [conv1, conv2])
    new_conv = nn.Conv2d(
        new_in_channels,
        new_out_channels,
        new_kernel,
        new_stride,
        new_pad,
        groups=conv1.in_channels if is_dw else 1,
    )
    return new_conv


def simulate_list_merge(convs: List[nn.Conv2d]):
    assert all(isinstance(x, nn.Conv2d) for x in convs)
    new_in_channels = convs[0].in_channels
    new_out_channels = convs[-1].out_channels
    new_kernel = convs[0].kernel_size[0]
    for conv in convs[1:]:
        new_kernel = (new_kernel // 2 + conv.kernel_size[0] // 2) * 2 + 1
    new_stride = convs[0].stride[0]
    for conv in convs[1:]:
        new_stride = new_stride * conv.stride[0]
    new_pad = convs[0].padding[0]
    for conv in convs[1:]:
        new_pad = new_pad + conv.padding[0]
    is_dw = all(x.groups == x.in_channels for x in convs)
    new_conv = nn.Conv2d(
        new_in_channels,
        new_out_channels,
        new_kernel,
        new_stride,
        new_pad,
        groups=convs[0].in_channels if is_dw else 1,
    )
    return new_conv


def trace_feat_size(blks, inp):
    pos, res = 0, dict()
    out = inp
    res[pos] = tuple(out.shape)
    for blk in blks:
        for lyr in blk.conv:
            out = lyr(out)
            if isinstance(lyr, nn.Conv2d):
                pos += 1
                res[pos] = tuple(out.shape)
    return res


def get_act(blocks, block_type):
    node_pos, act_pos = 0, set()
    for block in blocks:
        if isinstance(block, block_type):
            for layer in block.conv:
                if isinstance(layer, nn.Conv2d):
                    node_pos += 1
                elif isinstance(layer, DepShrinkReLU6):
                    if layer.act_hat:
                        act_pos.add(node_pos)
                elif isinstance(layer, (nn.ReLU, nn.ReLU6)):
                    act_pos.add(node_pos)
    act_num = node_pos
    return act_pos, act_num


# Reset the layers between `st_pos` and `end_pos`
def reset_layers(blocks, block_type, st_pos, end_pos, logger=None):
    node_pos = 0
    for block in blocks:
        if isinstance(block, block_type):
            for layer in block.conv:
                if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                    if isinstance(layer, nn.Conv2d):
                        node_pos += 1
                    if st_pos < node_pos and node_pos <= end_pos:
                        if logger:
                            logger.comment(str(layer))
                        layer.reset_parameters()


def get_blk(blocks, block_type):
    node_pos, blk_pos = 0, set()
    for block in blocks:
        if isinstance(block, block_type):
            for _, (_, layer) in enumerate(block.conv._modules.items()):
                if isinstance(layer, nn.Conv2d):
                    node_pos += 1
        blk_pos.add(node_pos)
    return blk_pos


def get_conv_lst(blocks, block_type, st, end):
    node_pos, lst = 0, []
    assert st < end
    for block in blocks:
        if isinstance(block, block_type):
            for layer in block.conv:
                if isinstance(layer, nn.Conv2d):
                    if node_pos >= st and node_pos < end:
                        lst.append(layer)
                    if node_pos + 1 == end:
                        return lst
                    node_pos += 1


def fix_act_lyrs(blocks, block_type, act_pos, act_type="relu6"):
    assert act_type in ["relu6", "relu"]
    node_pos = 0
    for block in blocks:
        if isinstance(block, block_type):
            for _, (name, layer) in enumerate(block.conv._modules.items()):
                if isinstance(layer, nn.Conv2d):
                    node_pos += 1
                elif isinstance(layer, (nn.ReLU, nn.ReLU6, nn.Identity)):
                    if node_pos in act_pos:
                        if act_type == "relu6":
                            setattr(block.conv, name, nn.ReLU6(inplace=True))
                        elif act_type == "relu":
                            setattr(block.conv, name, nn.ReLU(inplace=True))
                        else:
                            raise NotImplementedError("Not right activation")
                    else:
                        setattr(block.conv, name, nn.Identity())


def valid_blks(model):
    skip_s2t = model.skip_s2t
    str_pos = sorted(list(model.str_pos))
    act_num = model.get_act_info()[1]
    breaks = str_pos + [act_num]
    phase = 0
    b_pos = breaks[phase]
    skip_pos = b_pos
    blks = []
    for st in range(0, act_num):
        if st == b_pos:
            if b_pos == breaks[phase]:
                phase += 1
            b_pos = breaks[phase]
        if st == skip_pos:
            skip_pos = b_pos
        end = st + 1
        while end <= min(b_pos, skip_pos):
            blks.append((st, end))
            if end in skip_s2t:
                end = skip_s2t[end]
            else:
                end += 1
        if st in skip_s2t:
            skip_pos = skip_s2t[st]
    return blks


# necessary for backwards compatibility
class _DeprecatedConvBNAct(Conv2dNormActivation):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The ConvBNReLU/ConvBNActivation classes are deprecated since 0.12 and will be removed in 0.14. "
            "Use torchvision.ops.misc.Conv2dNormActivation instead.",
            FutureWarning,
        )
        if kwargs.get("norm_layer", None) is None:
            kwargs["norm_layer"] = nn.BatchNorm2d
        if kwargs.get("activation_layer", None) is None:
            kwargs["activation_layer"] = nn.ReLU6
        super().__init__(*args, **kwargs)


ConvBNReLU = _DeprecatedConvBNAct
ConvBNActivation = _DeprecatedConvBNAct


class DepShrinkReLU6(nn.ReLU6):
    # Inplace op not supported
    def __init__(self, inplace=False) -> None:
        super().__init__()
        self.act = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        # act_hat is either 1 or 0; indicates if activation is alive
        self.act_hat = 1.0

    def forward(self, x):
        # straight-forward estimator
        act = torch.clamp(self.act, 0, 1)
        act = self.act_hat + act - act.detach()
        return act * F.relu6(x) + (1 - act) * x

    def __repr__(self):
        if self.act_hat == 1.0:
            string = (
                f"{Fore.GREEN}DepShrinkReLU6() Enabled {Style.RESET_ALL}"
                + f"[Act : {self.act.data.item():.2e}]"
                + f"[Act Hat : {self.act_hat}]"
            )
        else:
            string = (
                f"{Fore.RED}DepShrinkReLU6() Disabled {Style.RESET_ALL}"
                + f"[Act : {self.act.data.item():.2e}] "
                + f"[Act Hat : {self.act_hat}]"
            )
        return string
