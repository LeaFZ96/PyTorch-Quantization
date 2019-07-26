import torch
import math

def linear_quantize(input, bits):
    
    range1 = input.max() - input.min()
    temp = (input - input.min()) / range1
    range2 = math.pow(2.0, bits) - 1
    min_v = - math.pow(2.0, bits - 1)
    ret = (temp.float() * range2) + min_v
    ret = ret.int().float()

    return ret