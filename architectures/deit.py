# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import os
import torch, torch.nn as nn
import pretrainedmodels as ptm

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


def select(arch, pretrained=True, **kwargs):
    if 'deit_tiny_patch16_224' in arch:
        return deit_tiny_patch16_224(pretrained=pretrained, **kwargs)
    elif 'deit_small_patch16_224' in arch:
        return deit_small_patch16_224(pretrained=pretrained, **kwargs)
    elif 'deit_base_patch16_224' in arch:
        return deit_base_patch16_224(pretrained=pretrained, **kwargs)
    elif 'deit_base_patch16_384' in arch:
        return deit_base_patch16_384(pretrained=pretrained, **kwargs)
    else:
        raise NotImplemented(f'Architecture {arch} has not been found.')

def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    embed_dim = 192

    if pretrained:
        path_pretrained = '/architectures/pretrained_weights/deit_tiny_patch16_224-a1311bcf.pth'
        if os.path.exists(path_pretrained):
            checkpoint = torch.load(path_pretrained, map_location="cpu")
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])

    return model, embed_dim

def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    embed_dim = 384

    if pretrained:
        path_pretrained = '/architectures/pretrained_weights/deit_small_patch16_224-cd65a155.pth'
        if os.path.exists(path_pretrained):
            checkpoint = torch.load(path_pretrained, map_location="cpu")
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model, embed_dim

def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    embed_dim = 768

    if pretrained:
        path_pretrained = '/architectures/pretrained_weights/deit_base_patch16_224-b5f2ef4d.pth'
        if os.path.exists(path_pretrained):
            checkpoint = torch.load(path_pretrained, map_location="cpu")
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model, embed_dim

def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    embed_dim = 768

    if pretrained:
        path_pretrained = '/architectures/pretrained_weights/deit_base_patch16_384-8de9b5d1.pth'
        if os.path.exists(path_pretrained):
            checkpoint = torch.load(path_pretrained, map_location="cpu")
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])

    return model, embed_dim


"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim):
        super(Network, self).__init__()

        self.arch  = arch
        self.embed_dim = embed_dim
        self.name = self.arch
        self.model, embed_dim_model = select(arch, pretrained=True if pretraining is not None else False)
        self.model.last_linear = torch.nn.Linear(embed_dim_model, embed_dim)

        print(f'Architecture:\ntype: {self.arch}\nembed_dims: {self.embed_dim}')


    def forward(self, x):
        x = self.forward_features(x)
        z = self.model.last_linear(x)

        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        return {'embeds':z}
