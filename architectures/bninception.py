"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import pretrainedmodels as ptm



"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim):
        super(Network, self).__init__()

        self.arch  = arch
        self.embed_dim = embed_dim
        self.name = self.arch
        self.model = ptm.__dict__['bninception'](num_classes=1000, pretrained=pretraining)
        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, embed_dim)
        if '_he' in self.arch:
            torch.nn.init.kaiming_normal_(self.model.last_linear.weight, mode='fan_out')
            torch.nn.init.constant_(self.model.last_linear.bias, 0)

        if 'frozen' in self.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.pool_base = F.avg_pool2d
        self.pool_aux  = F.max_pool2d if 'double' in self.arch else None

        print(f'ARCHITECTURE:\ntype: {self.arch}\nembed_dims: {self.embed_dim}\n')

    def forward(self, x, warmup=False, **kwargs):
        x = self.model.features(x)
        prepool_y = y = self.pool_base(x,kernel_size=x.shape[-1])
        if self.pool_aux is not None:
            y += self.pool_aux(x, kernel_size=x.shape[-1])
        if 'lp2' in self.arch:
            y += F.lp_pool2d(x, 2, kernel_size=x.shape[-1])
        if 'lp3' in self.arch:
            y += F.lp_pool2d(x, 3, kernel_size=x.shape[-1])

        y = y.view(len(x),-1)
        if warmup:
            x,y,prepool_y = x.detach(), y.detach(), prepool_y.detach()
        
        z = self.model.last_linear(y)
        if 'normalize' in self.name:
            z = F.normalize(z, dim=-1)

        return {'embeds':z, 'avg_features':y, 'features':x, 'extra_embeds': prepool_y}
