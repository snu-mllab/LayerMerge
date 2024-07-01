import torch
import torch.nn as nn
import numpy as np
import time
import sys

try:
    import torch_tensorrt
    from torch_tensorrt.logging import set_reportable_log_level, Level
except:
    print("TensorRT is not installed")


@torch.no_grad()
def get_time(
    model, input_shape=(1, 3, 224, 224), name="", rep=1000, warmup=2000, verb=False
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
    print(f"[{name:>20}] MEAN : {mean_syn:.2f}ms || STD : {std_syn:.2f}")
    print(f"[{name:>20}] THPT : {mean_thpt:.2f} || STD : {std_thpt:.2f}")
    return mean_syn, std_syn


def unroll_merged(module):
    result = nn.Sequential()
    for ind, blk in enumerate(module.m_features):
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
def compile_and_time(module, sz, txt, verb=False):
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
        model, input_shape=sz, name=txt, rep=30, warmup=10, verb=verb
    )
    del module
    return result, std


@torch.no_grad()
def torch_time(module, sz, txt, verb=False, rep=100, warmup=100):
    module.eval()
    print(f"Measuring time without TensorRT compiling...")
    result, std = get_time(
        module, input_shape=sz, name=txt, rep=rep, warmup=warmup, verb=verb
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
