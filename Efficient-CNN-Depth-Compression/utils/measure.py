import torch
import torch.nn as nn
import numpy as np
import time
import sys
from einops import repeat

try:
    import torch_tensorrt
    from torch_tensorrt.logging import set_reportable_log_level, Level
except:
    print("TensorRT is not installed")


@torch.no_grad()
def get_time(
    model,
    input_shape=(1, 3, 224, 224),
    name="",
    rep=1000,
    warmup=2000,
    verb=False,
    logger=None,
):
    st = time.time()
    device = torch.device("cuda")
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    timings = np.zeros((rep, 1))
    # GPU-WARM-UP
    for _ in range(warmup):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep_ in range(rep):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep_] = curr_time
    mean_syn = np.sum(timings) / rep
    std_syn = np.std(timings)
    batch_size = input_shape[0]
    thpts = 1 / (timings * 0.001) * batch_size
    mean_thpt = np.sum(thpts) / rep
    std_thpt = np.std(thpts)
    if verb:
        print(f"Batch size is {batch_size}")
        print(f"Measure time {time.time() - st:.2f} seconds")
    print(f"[{name:>20}] THPT : {mean_thpt:.2f} || STD : {std_thpt:.2f}")
    print(f"[{name:>20}] MEAN : {mean_syn:.2f}ms || STD : {std_syn:.2f}")
    if logger:
        logger.comment(f"[{name:>20}] THPT : {mean_thpt:.2f} || STD : {std_thpt:.2f}")
        logger.comment(f"[{name:>20}] MEAN : {mean_syn:.2f}ms || STD : {std_syn:.2f}")
    return mean_syn, std_syn


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


def fuse_conv_bn(conv, bn):
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
    return new_conv


def unroll_merged(module):
    module.to("cpu")
    result = nn.Sequential()
    for ind, blk in enumerate(module.m_features):
        if isinstance(blk, nn.Sequential):
            blk = nn.Sequential(fuse_conv_bn(blk[0], blk[1]), blk[2])
        result.add_module(f"blk{ind}", blk)

    pre_last = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
    for ind, blk in enumerate(pre_last):
        result.add_module(f"pre_last{ind}", blk)

    for ind, blk in enumerate(module.classifier):
        result.add_module(f"last{ind}", blk)
    return result


def unroll_merged_vgg(module):
    result = nn.Sequential()
    for ind, blk in enumerate(module.m_features):
        result.add_module(f"blk{ind}", blk)

    pre_last = [nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten()]
    for ind, blk in enumerate(pre_last):
        result.add_module(f"pre_last{ind}", blk)

    for ind, blk in enumerate(module.classifier):
        result.add_module(f"last{ind}", blk)
    return result


@torch.no_grad()
def compile_and_time(module, sz, txt, verb=False, logger=None):
    if not "torch_tensorrt" in sys.modules:
        raise Exception("You should install TensorRT")
    set_reportable_log_level(Level.Error)
    st = time.time()
    module.eval()
    model = torch_tensorrt.compile(
        module,
        inputs=[torch_tensorrt.Input(sz)],
        enabled_precisions={torch_tensorrt.dtype.float},
    )
    print(f"TensorRT compiling done ({time.time() - st:.2f} seconds)")
    result, std = get_time(
        model, input_shape=sz, name=txt, rep=100, warmup=200, verb=verb, logger=logger
    )
    del module
    return result, std


@torch.no_grad()
def torch_time(module, sz, txt, verb=False, rep=100, warmup=200, logger=None):
    module.eval()
    print(f"Measuring time without TensorRT compiling...")
    result, std = get_time(
        module,
        input_shape=sz,
        name=txt,
        rep=rep,
        warmup=warmup,
        verb=verb,
        logger=logger,
    )
    del module
    return result, std


@torch.no_grad()
def get_cpu_time(
    model, input_shape=(1, 3, 224, 224), name="", rep=10, warmup=100, verb=False
):
    st = time.time()
    device = torch.device("cpu")
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    # INIT LOGGERS
    timings = np.zeros((rep, 1))
    # CPU-WARM-UP
    for i in range(warmup):
        _ = model(dummy_input)
        if i % 10 == 0:
            print(f"{i} / {warmup}")
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep_ in range(rep):
            start = time.time() * 1000
            _ = model(dummy_input)
            end = time.time() * 1000
            curr_time = end - start
            timings[rep_] = curr_time
            if rep_ % 10 == 0:
                print(f"{rep_} / {rep}")
    mean_syn = np.sum(timings) / rep
    std_syn = np.std(timings)
    batch_size = input_shape[0]
    thpts = 1 / (timings * 0.001) * batch_size
    mean_thpt = np.sum(thpts) / rep
    std_thpt = np.std(thpts)
    if verb:
        print(f"Batch size is {batch_size}")
        print(f"Measure time {time.time() - st:.2f} seconds")
        print(f"[{name:>20}] THPT : {mean_thpt:.2f} || STD : {std_thpt:.2f}")
    print(f"[{name:>20}] MEAN : {mean_syn:.2f}ms || STD : {std_syn:.2f}")
    return mean_syn, std_syn


@torch.no_grad()
def torch_cpu_time(module, sz, txt, verb=False):
    module.eval()
    print(f"Measuring time with cpu...")
    result, std = get_cpu_time(
        module, input_shape=sz, name=txt, rep=100, warmup=10, verb=verb
    )
    del module
    return result, std
