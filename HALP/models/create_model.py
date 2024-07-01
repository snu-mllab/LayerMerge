# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
#
# Official PyTorch implementation of NeurIPS2022 paper
# Structural Pruning via Latency-Saliency Knapsack
# Maying Shen, Hongxu Yin, Pavlo Molchanov, Lei Mao, Jianna Liu and Jose M. Alvarez
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# --------------------------------------------------------

from models.resnet import make_resnet
from models.resnet_fused import fuse_resnet
from models.resnet_pruned import make_pruned_resnet
from layer_merge.models.resnet_merged import make_depth_resnet
from layer_merge.models.resnet_merged_layer import make_depth_layer_resnet
from layer_merge.models.resnet_layer import make_layer_resnet

import torch


def fuse_model(arch, model):
    if "resnet" in arch.lower():
        model = fuse_resnet(model)
    else:
        raise NotImplementedError
    return model

def get_model(
    arch,
    class_num,
    enable_bias,
    group_mask_file=None,
    depth_file=None,
    depth_method=None,
):
    if not group_mask_file is None:
        if "resnet" in arch.lower():
            model = make_pruned_resnet(arch, group_mask_file, class_num, enable_bias)
        else:
            raise NotImplementedError
    elif not depth_file is None:
        assert depth_method != None
        state = torch.load(depth_file)
        if "resnet" in arch.lower():
            if depth_method == "kim23efficient":
                act_pos, merge_pos = state["act_pos"], state["merge_pos"]
                model = make_depth_resnet(
                    arch, class_num, enable_bias, act_pos, merge_pos
                )
            elif depth_method == "kim24layermerge":
                act_ind, conv_ind = state["act_ind"], state["conv_ind"]
                model = make_depth_layer_resnet(
                    arch, class_num, enable_bias, act_ind, conv_ind
                )
            elif depth_method == "kim24layer":
                conv_ind = state["conv_ind"]
                model = make_layer_resnet(
                    arch, class_num, enable_bias, conv_ind
                )
        else:
            raise NotImplementedError
    else:
        if "resnet" in arch.lower():
            model = make_resnet(arch, class_num, enable_bias)
        else:
            raise NotImplementedError

    return model
