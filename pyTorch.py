# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
import torch
import numpy as np
import pandas as pd

def get_device(use_cuda = True):
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    return device

def load_to_device(device, *arg):
    """ Transfer all items in arg to torch.device """
    rt = []
    for each_item in arg:
        if isinstance(each_item, np.ndarray):
            rt.append(torch.from_numpy(each_item).to(device, dtype=torch.float32))
        elif isinstance(each_item, torch.Tensor):
            rt.append(each_item.to(device, dtype=torch.float32))
    return rt
    
def norm_01_tensor(vector):
    min_v = torch.min(vector)
    range_v = torch.max(vector) - min_v
    if range_v > 0:
        normalised = (vector - min_v) / range_v
    else:
        normalised = torch.zeros(vector.size())
    return normalised

def norm_01_np(array):
    min_v = np.min(array)
    range_v = np.max(array) - min_v
    if range_v > 0:
        normalised = (array - min_v) / range_v
    else:
        normalised = np.zeros(array.size())
    return normalised

def norm_01(*arg, low_lim = 0, high_lim = 1):
    scale_range = high_lim - low_lim
    rt = []
    for each_item in arg:
        if isinstance(each_item, np.ndarray):
            rt.append(norm_01_np(each_item))
        elif isinstance(each_item, torch.Tensor):
            rt.append(norm_01_tensor(each_item))
    if len(rt) == 1:
        return rt[0]
    else:
        return rt

def force_convert(*arg, to_tensor = True):
    rt = []
    if to_tensor:
        for each_item in arg:
            if isinstance(each_item, torch.Tensor):
                rt.append(each_item)
            elif isinstance(each_item, np.array):
                rt.append(torch.from_numpy(each_item))
            elif isinstance(each_item, pd.DataFrame):
                rt.append(torch.from_numpy(each_item.values))
            else:
                print("Unknown TypeError")
    else:
        for each_item in arg:
            if isinstance(each_item, torch.tensor):
                rt.append(each_item.cpu().numpy())
            elif isinstance(each_item, np.array):
                rt.append(each_item)
            else:
                print("Whats that?")
    return rt
    
    
    