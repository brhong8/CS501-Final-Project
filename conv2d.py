import torch
import torch.nn as nn
import sys
import argparse

def run_convolution(kernel_size, stride, height, width, dtype, device):
    conv2d = nn.Conv2d(2, 28, kernel_size, stride=stride, device=device, dtype=dtype)

    input = torch.randn(2, height, width, device=device, dtype=dtype)

    output = conv2d(input)

    return output

if __name__ == "__main__":
    if torch.cuda.is_available:
        device=torch.device('cuda:0')
    else:
        device='cpu'

    parser = argparse.ArgumentParser()

    dtypes = ['fp32', 'fp16', 'fp64']

    parser.add_argument('--kernel_size', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--height', type=int, default=50)
    parser.add_argument('--width', type=int, default=50)
    parser.add_argument('--dtype', type=str, choices=dtypes, default='fp64')

    args = parser.parse_args()

    kernel_size = args.kernel_size
    stride = args.stride
    height = args.height
    width = args.width
    match args.dtype:
        case 'fp16':
            dtype = torch.half
        case 'fp32':
            dtype = torch.float
        case 'fp64':
            dtype = torch.double

    run_convolution(kernel_size, stride, height, width, dtype, device)
    