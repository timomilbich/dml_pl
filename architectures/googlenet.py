"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import torchvision.models as mod





"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch):
        super(Network, self).__init__()

        self.arch  = arch
        self.model = mod.googlenet(pretrained=True)

        self.model.last_linear = torch.nn.Linear(self.model.fc.in_features, opt.embed_dim)
        self.model.fc = self.model.last_linear

        self.name = self.arch

    def forward(self, x):
        x = self.model(x)
        if not 'normalize' in self.arch:
            return x
        return torch.nn.functional.normalize(x, dim=-1)
