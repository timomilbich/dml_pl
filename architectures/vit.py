from functools import partial
import os
import torch, torch.nn as nn

import timm.models.vision_transformer as timm_vit
from timm.models.vision_transformer import _create_vision_transformer, _cfg


def select_model(arch, pretrained=True, **kwargs):
    if 'vit_base_patch32_224' in arch:
        return timm_vit.vit_base_patch32_224_in21k(pretrained=pretrained, **kwargs), 768 # CHECK IF NEW TIMM PIP IS AVAILABLE TO USE NORMAL IMAGENET
    elif 'vit_base_patch16_224_in21k' in arch:
        return timm_vit.vit_base_patch16_224_in21k(pretrained=pretrained, **kwargs), 768 # CHECK IF NEW TIMM PIP IS AVAILABLE TO USE NORMAL IMAGENET
    elif 'vit_base_patch16_224' in arch:
        return timm_vit.vit_base_patch16_224(pretrained=pretrained, **kwargs), 768 # CHECK IF NEW TIMM PIP IS AVAILABLE TO USE NORMAL IMAGENET
    elif 'vit_small_patch16_224_in21k' in arch:
        model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
        cfg_custom = _cfg(url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz', num_classes=21843),
        model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, default_cfg=cfg_custom[0], **model_kwargs)
        return model, 384
    elif 'vit_small_patch16_224' in arch:
        model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
        cfg_custom = _cfg(url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz')
        model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, default_cfg=cfg_custom, **model_kwargs)
        return model, 384
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
