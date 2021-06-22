from functools import partial
import os
import torch, torch.nn as nn

import timm.models.vision_transformer as timm_vit


def select_model(arch, pretrained=True, **kwargs):
    if 'vit_base_patch32_224' in arch:
        return timm_vit.vit_base_patch32_224_in21k(pretrained=pretrained, **kwargs), 768 # CHECK IF NEW TIMM PIP IS AVAILABLE TO USE NORMAL IMAGENET
    else:
        raise NotImplemented(f'Architecture {arch} has not been found.')


"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim):
        super(Network, self).__init__()

        self.arch  = arch
        self.embed_dim = embed_dim
        self.name = self.arch
        self.features, embed_dim_model = select_model(arch, pretrained=True if pretraining is not None else False)
        self.last_linear = torch.nn.Linear(embed_dim_model, embed_dim) if embed_dim > 0 else nn.Identity()

        print(f'Architecture:\ntype: {self.arch}\nembed_dims: {self.embed_dim}')


    def forward(self, x):
        x = self.features.forward_features(x)
        z = self.last_linear(x)

        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        return {'embeds':z}
