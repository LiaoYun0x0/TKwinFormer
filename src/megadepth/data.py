import os
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader
)

from src.megadepth.megadepth import MegaDepthDataset
from src.megadepth.sampler import RandomConcatSampler

def build_dataset(root_dir,list_root,list_path,mode,img_resize,score=0.4,max_data_size=1e8,n_samples_per_subset=100):
    with open(list_path,'r') as txt:
        npz_names = txt.readlines()
    datasets = []
    count_size = 0
    for npz_name in npz_names:
        if count_size > max_data_size:
            break
        npz_name = npz_name.strip('\n')
        if not npz_name.endswith('.npz'):
            npz_name += '.npz'
        npz_path = os.path.join(list_root,npz_name)
        dataset = MegaDepthDataset(root_dir, npz_path, mode, 
                                        min_overlap_score=score,
                                        img_resize=img_resize, img_padding=True,depth_padding=True,df=8)
        
        datasets.append(dataset)
        curr_size = len(datasets[-1])
        if count_size + curr_size <= max_data_size:
            count_size += curr_size
        else:
            load_size = int(max_data_size - count_size)
            count_size += load_size
            datasets[-1].pair_infos = datasets[-1].pair_infos[:load_size]
            break
    dataset = ConcatDataset(datasets)
    sampler = RandomConcatSampler(dataset,n_samples_per_subset,True,True,1,66)
    return dataset,sampler

def build_test_dataset(root_dir,list_root,list_path,mode,img_resize,score=0.4,max_data_size=1e8):
    with open(list_path,'r') as txt:
        npz_names = txt.readlines()
    datasets = []
    count_size = 0
    for npz_name in npz_names:
        if count_size > max_data_size:
            break
        npz_name = npz_name.strip('\n')
        if not npz_name.endswith('.npz'):
            npz_name += '.npz'
        npz_path = os.path.join(list_root,npz_name)
        dataset = MegaDepthDataset(root_dir, npz_path, mode, 
                                        min_overlap_score=score,
                                        img_resize=img_resize, img_padding=True,depth_padding=True,df=8)
        
        datasets.append(dataset)
        curr_size = len(datasets[-1])
        if count_size + curr_size <= max_data_size:
            count_size += curr_size
        else:
            load_size = int(max_data_size - count_size)
            count_size += load_size
            datasets[-1].pair_infos = datasets[-1].pair_infos[:load_size]
            break
    dataset = ConcatDataset(datasets)
    return dataset