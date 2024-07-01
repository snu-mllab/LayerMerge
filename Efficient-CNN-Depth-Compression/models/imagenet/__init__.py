from .mobilenetv2 import *
from .mobilenetv2_com import *
from .mobilenetv2_ds import *
from .vgg import *
from .vgg_com import *

models = {
    "mobilenet_v2": mobilenet_v2,
    "learn_mobilenet_v2": learn_mobilenet_v2,
    "dep_shrink_mobilenet_v2": dep_shrink_mobilenet_v2,
    "vgg19": vgg19_bn,
    "learn_vgg19": learn_vgg19_bn,
}

blocks = {
    "mobilenet_v2": InvertedResidual,
    "learn_mobilenet_v2": InvertedResidual,
    "dep_shrink_mobilenet_v2": InvertedResidual,
    "vgg19": VGGBlock,
    "learn_vgg19": LearnVGGBlock,
}
