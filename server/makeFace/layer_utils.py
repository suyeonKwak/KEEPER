import torch
import torch.nn as nn
import numpy as np


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(
            -1, -1, -1, factor, -1, factor
        )
        x = x.contiguous().view(
            shape[0], shape[1], factor * shape[2], factor * shape[3]
        )
    return x
