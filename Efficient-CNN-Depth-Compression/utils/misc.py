import os
import shutil

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


def save_checkpoint(state, is_best, checkpoint="checkpoint", filename="checkpoint.pth"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = filepath[:-4]
        best_filepath += "_best.pth"
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    model,
    arch,
    path,
    act_path=None,
    ds_pat=None,
    logger=None,
):

    if os.path.isfile(path):
        source_state = torch.load(path)
        log_tool(f"=> loading pretrained weight '{path}'", logger)
        if "epoch" in source_state:
            log_tool(f"=> (epoch {source_state['epoch']})", logger)
    else:
        log_tool(f"=> no weight found at '{path}'", logger)
        exit()

    if act_path == None:
        act_path = path

    if arch in ["mobilenet_v2", "vgg19"]:
        log_tool(f"=> loading on the architecture '{arch}'", logger)
        act_state = dict()
    elif os.path.isfile(act_path):
        act_state = torch.load(act_path)
        log_tool(f"=> loading activations from '{act_path}'", logger)
        if "act_pos" in act_state:
            log_tool(f"Number of activations : {len(act_state['act_pos'])}", logger)
            log_tool(f"{act_state['act_pos']}", logger)
        else:
            log_tool(f"=> no activations found at '{path}'", logger)
    else:
        log_tool(f"=> not a valid act_path '{act_path}'", logger)
        exit()

    if ds_pat != None:
        ds_pattern, compress_k = ds_pat
        # fmt: off
        pat2cmp = {
            "A": 11, "B": 8, "C": 8, "D": 6, "E": 6, "F": 0, "A10": 12, "B10": 12, "C10": 9, "D10": 7, "AR": 11, "BR": 8, "CR": 6, "AR10": 12, "BR10": 9, "CR10": 7, "AR_AUG": 11, "BR_AUG": 8, "CR_AUG": 6, "AR10_AUG": 12, "BR10_AUG": 9, "CR10_AUG": 7
        }
        # fmt: on
        assert ds_pattern != "none" and arch == "dep_shrink_mobilenet_v2"
        assert compress_k == pat2cmp[ds_pattern]
        log_tool(f"=> training from ds_pattern '{ds_pattern}'", logger)
        source_state["compress_k"] = compress_k
        model.module.load_pattern(ds_pattern)

    if arch == "dep_shrink_mobilenet_v2":
        model.module.compress_k = source_state["compress_k"]

    # `act_pos` for finetuned/merged network weights
    if "act_pos" in act_state:
        a_pos = act_state["act_pos"]
        source_state["act_pos"] = a_pos

        # `merge_pos` for merged network weights
        if "merge_pos" in act_state:
            log_tool(f"=> loading optimal merge pattern from '{act_path}'", logger)
            log_tool(f"{act_state['merge_pos']}", logger)
            m_pos = act_state["merge_pos"]
            source_state["merge_pos"] = m_pos
        else:
            m_pos = None

        if arch in ["learn_mobilenet_v2", "learn_vgg19"]:
            model.module.fix_act(a_pos, m_pos)
        else:
            model.module.fix_act(a_pos)

        if "merged" in act_state:
            model.to("cpu")
            if m_pos:
                assert arch in ["learn_mobilenet_v2", "learn_vgg19"]
                model.module.merge(act_pos=act_state["act_pos"], merge_pos=m_pos)
            else:
                model.module.merge(act_pos=act_state["act_pos"])
            model.to("cuda")

    model.load_state_dict(source_state["state_dict"], strict=False)

    del source_state["state_dict"]
    return source_state


def cp_state(new_state, source_state, name):
    if name in source_state:
        new_state[name] = source_state[name]


def print_act_pos(module, source_state):
    with torch.no_grad():
        if "act_pos" in source_state:
            act_pos = source_state["act_pos"]
            print("learned activation is below")
            print(sorted(list(act_pos)))
            print(f"{len(act_pos)} alive")


def log_tool(string, logger, mode="print"):
    if mode == "print":
        print(string)
        if logger != None:
            logger.comment(f"{string}")
    elif mode == "opt":
        logger.comment("\n-----------------------------\n")
        logger.comment(string)
        logger.comment("\n-----------------------------\n")
    elif mode == "ap":
        logger.comment("-----------------------------")
        logger.comment("learned activation is below")
        logger.comment(string)
        logger.comment("-----------------------------")


# Implementation adapted from AlphaNet - https://github.com/facebookresearch/AlphaNet
class KLLossSoft(torch.nn.modules.loss._Loss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    T: temperature
    """

    def __init__(
        self, alpha, teacher, size_average=None, reduce=None, reduction: str = "mean"
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.alpha = alpha
        self.teacher = teacher

    def forward(self, output, soft_logits, target, temperature=1.0):
        output, soft_logits = output / temperature, soft_logits / temperature
        soft_target_prob = torch.nn.functional.softmax(soft_logits, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        kd_loss = -torch.sum(soft_target_prob * output_log_prob, dim=1)
        if target is not None:
            target = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
            target = target.unsqueeze(1)
            output_log_prob = output_log_prob.unsqueeze(2)
            ce_loss = -torch.bmm(target, output_log_prob).squeeze()
            loss = (
                self.alpha * temperature * temperature * kd_loss
                + (1.0 - self.alpha) * ce_loss
            )
        else:
            loss = kd_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
