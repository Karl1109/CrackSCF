from thop import profile
import torch
from main import get_args_parser
from models import build_model
import argparse
from util.misc import NestedTensor

parser = argparse.ArgumentParser('CrackSCF', parents=[get_args_parser()])
args = parser.parse_args()

if __name__ == '__main__':
    model, _ = build_model(args)
    model.to(args.device)

    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total Parameters: {total_params}")

    input = torch.randn(1, 3, 384, 384)
    input_shape = input.shape
    dim1 = input_shape[0]
    dim2 = input_shape[2]
    dim3 = input_shape[3]
    mask = torch.zeros_like(torch.randn(dim1, dim2, dim3))
    mask = mask.bool()

    samples = NestedTensor(input, mask).to(torch.device(args.device))
    flops, params = profile(model, (samples, ))
    print("flops(G):", flops/1e9, "params(M):", params/1e6)
