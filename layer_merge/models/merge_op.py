import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from einops import rearrange, repeat


class NaiveFeed(nn.Module):
    def __init__(self, odict: OrderedDict) -> None:
        super().__init__()
        self.md = nn.Sequential(odict)

    def forward(self, x):
        return self.md(x)


class SkipFeed(nn.Module):
    def __init__(self, odict: OrderedDict, last=nn.Identity) -> None:
        super().__init__()
        self.md = nn.Sequential(odict)
        self.last = last()

    def forward(self, x):
        return self.last(self.md(x) + x)


class Downsample(nn.Module):
    def __init__(self, planes) -> None:
        super().__init__()
        self.planes = planes

    def forward(self, x):
        sz = x.shape[3] // 2
        ch = x.shape[1] // 2
        out = x
        out = F.interpolate(out, size=(sz, sz))
        zeros = out.mul(0)
        out = torch.cat((zeros[:, :ch, :, :], out), 1)
        out = torch.cat((out, zeros[:, ch:, :, :]), 1)
        return out


class SkipFeedDown(nn.Module):
    def __init__(
        self, odict: OrderedDict, last=nn.Identity, downsample=nn.Identity()
    ) -> None:
        super().__init__()
        self.md = nn.Sequential(odict)
        self.last = last()
        self.downsample = downsample

    def forward(self, x):
        return self.last(self.md(x) + self.downsample(x))


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

def adjust_with_bn_linear(weight, bias, bn):
    # gamma, beta, running_mean, running_var
    bw, bb, brm, brv = unroll_bn_params(bn)
    for i in range(bb.size(0)):
        bias[i] = bias[i] / torch.sqrt(brv[i] + 1e-5) * bw[i]
        weight[i] = weight[i] / torch.sqrt(brv[i] + 1e-5) * bw[i]

    return weight, bias

def conv_with_new(old_conv, old_bias, new_conv, stride, padding):
    old_conv = rearrange(old_conv, "oup inp h w -> inp oup h w")
    f_new_conv = torch.flip(new_conv, [2, 3])
    old_conv = F.conv2d(
        input=old_conv, weight=f_new_conv, stride=1, padding=f_new_conv.shape[2] - 1
    )
    old_conv = rearrange(old_conv, "inp oup h w -> oup inp h w")
    old_bias = torch.einsum("jikl,i->j", new_conv, old_bias)
    return old_conv, old_bias


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


def unroll_lyrs(pos, weights, lyrs):
    isize, m_cw, m_p, m_b = weights
    conv, bn = lyrs
    pos += 1

    cw, bw = unroll_conv_params(conv, conv.in_channels == conv.groups)
    # feature size after conv
    isize = (isize - cw.size(2) + 2 * conv.padding[0]) // conv.stride[0] + 1
    m_p = conv.padding[0]
    m_cw = cw
    m_b = torch.zeros(cw.size(0))
    if bw != None:
        m_b += bw

    assert isinstance(bn, nn.BatchNorm2d) or bn == None
    if isinstance(bn, nn.BatchNorm2d):
        m_cw, m_b = adjust_with_bn(m_cw, m_b, bn)
    
    ctype = "dw" if conv.in_channels == conv.groups else "cw"
    cstr = conv.stride[0]

    return pos, (isize, m_cw, m_p, m_b), (ctype, cstr)


def fuse_skip(cw):
    # Fuse identity addition
    mid = cw.size(2) // 2
    for i in range(cw.size(1)):
        cw[i][i][mid][mid] += 1


def push_merged_layers(m_layers, pos, weights, cparam, relu=False, act_type="relu"):
    assert act_type in ["relu6", "relu"]
    _, m_cw, m_p, m_b = weights
    ctype, cstr = cparam
    push_layer(m_layers, f"merged_conv{pos}", m_cw, ctype, cstr, m_p, m_b)
    if relu:
        push_layer(m_layers, f"relu{pos}", None, act_type)


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


# Reset the layers between `st_pos` and `end_pos`
def reset_layers(convs, bns, st_pos, end_pos):
    for node_pos in convs.keys():
        conv, bn = convs[node_pos][1], bns[node_pos][1]
        if st_pos < node_pos and node_pos <= end_pos:
            print(str(conv))
            print(str(bn))
            conv.reset_parameters()
            bn.reset_parameters()

def reset_convs(convs, st_pos, end_pos):
    for node_pos in convs.keys():
        conv = convs[node_pos][1]
        if st_pos < node_pos and node_pos <= end_pos:
            print(str(conv))
            conv.reset_parameters()

def identity_conv_bn(in_channels, dw=True):
    # Depthwise Convolution with kernel size 1
    if dw:
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
    else:
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    # Set the parameters to make it an identity mapping
    conv.weight.data.fill_(0)
    if dw:
        for i in range(in_channels):
            conv.weight.data[i, 0, 0, 0] = 1
    else:
        for i in range(in_channels):
            conv.weight.data[i, i, 0, 0] = 1

    # Batch Normalization
    bn = nn.BatchNorm2d(in_channels)

    # Set the parameters to make it an identity mapping
    bn.weight.data.fill_(1)  # gamma = 1
    bn.bias.data.fill_(0)    # beta = 0
    bn.running_mean.fill_(0) # running mean = 0
    bn.running_var.fill_(1-1e-5)  # running variance = 1

    conv.eval()
    bn.eval()

    return conv, bn

def l1_norm_sum(layer):
    """ Compute the L1 norm sum of the weights of a convolutional layer. """
    return torch.sum(torch.abs(layer.weight.data)).item()

def find_max_importance_candidate(conv_layers, candidates):
    """
    Find the candidate index with the largest importance.

    Parameters:
    conv_layers (list of nn.Module): List of convolutional layers.
    candidates (list of tuple): List of candidate indices.

    Returns:
    tuple: The candidate index with the largest importance.
    """
    max_importance = -float('inf')
    max_index = None

    for idx in candidates:
        current_importance = sum(l1_norm_sum(conv_layers[i]) for i in idx)

        if current_importance > max_importance:
            max_importance = current_importance
            max_index = idx

    return max_index
