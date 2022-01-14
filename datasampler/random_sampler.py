import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random
import torch.distributed as dist



"""======================================================"""
REQUIRES_STORAGE = False

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, image_dict, image_list, batch_size, drop_last=False, num_replicas=None, rank=None):

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                num_replicas = dist.get_world_size()
            except:
                num_replicas = 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                rank = dist.get_rank()
            except:
                rank = 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        #####
        self.image_dict = image_dict
        self.image_list = image_list

        #####
        self.train_classes = list(self.image_dict.keys())

        ####
        self.batch_size = batch_size
        self.sampler_length_total = len(image_list) // batch_size
        self.sampler_length = self.sampler_length_total // self.num_replicas

        self.name = 'random_sampler'
        self.requires_storage = False

        print(f"\nData sampler [{self.name}] initialized with rank=[{self.rank}/{self.num_replicas}] and sampler length=[{self.sampler_length}/{self.sampler_length_total}].\n")


    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            for _ in range(self.batch_size-1):
                class_key  = random.choice(list(self.image_dict.keys()))
                sample_idx = np.random.choice(len(self.image_dict[class_key]))
                subset.append(self.image_dict[class_key][sample_idx][-1])
            #
            subset.append(random.choice(self.image_dict[self.image_list[random.choice(subset)][-1]])[-1])
            yield subset

    def __len__(self):
        return self.sampler_length
