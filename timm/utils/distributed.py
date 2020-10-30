""" Distributed training/validation utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import distributed as dist

from .model import unwrap_model


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)

def gather_same_shaped_tensor(tensor, n):
    return_list = [torch.zeros_like(tensor) for _ in range(n)]
    dist.all_gather(return_list, tensor)
    return return_list

def gather_tensor(tensor, n):
    size = torch.tensor([tensor.shape[0]], dtype=torch.int64).cuda()
    size = torch.stack(gather_same_shaped_tensor(size, n)).cpu()
    max_size = max(size)
    
    padded_tensor = torch.empty(
        max_size, *tensor.shape[1:],
        dtype=tensor.dtype,
    ).cuda()
    
    padded_tensor[:tensor.shape[0]] = tensor
    gathered_tensor = gather_same_shaped_tensor(padded_tensor, n)
    stacked_tensor = torch.cat([tensor[:shape] for tensor, shape in zip(gathered_tensor, size)])
    
    return stacked_tensor
