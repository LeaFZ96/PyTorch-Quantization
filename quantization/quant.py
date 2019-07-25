import torch
import math

def linear_quantize(input, bits):

    """assert bits >= bits, bits
    if bits == 1:
        return torch.sign(input) - 1
    bound = math.pow(2.0, bits - 1)
    min_val = - bound
    max_val = bound - 1
    rounded = input // 1
    result = torch.clamp(rounded, min_val, max_val)"""
    temp = input * 127
    temp = temp.char()
    result = temp.half()
    return result