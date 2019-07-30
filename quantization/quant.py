import torch
from torch.autograd import Variable
import math

def linear_quantize(input, bits):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    # print('sorted_value: ', sorted_value)
    split_idx = 0
    v = sorted_value[split_idx]
    # print('v: ', v)
    sf = math.ceil(math.log2(v+1e-12))

    sf = bits - 1. - sf

    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    # rounded = torch.floor(input / delta + 0.5)
    rounded = (input / delta + 0.5) // 1

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

"""def linear_quantize(input, bits):
    
    range1 = input.max() - input.min()
    temp = (input - input.min()) / range1
    range2 = math.pow(2.0, bits) - 1
    min_v = - math.pow(2.0, bits - 1)
    ret = (temp.float() * range2) + min_v
    ret = ret.float()

    return ret"""