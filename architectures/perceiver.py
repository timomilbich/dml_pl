"""
The network architectures and weights are adapted and used from https://github.com/lucidrains/perceiver-pytorch and
https://github.com/deepmind/deepmind-research/tree/master/perceiver#colabs
"""
import torch
from perceiver_pytorch import Perceiver
from perceiver_pytorch import PerceiverIO

DEFAULT_PERCEIVER_PARAMS = {
    'depth': 6,  # depth of net. The shape of the final attention mechanism will be: depth * (cross attention -> self_per_cross_attn * self attention)
    'num_latents': 256, # number of latents, or induced set points, or centroids. different papers giving it different names
    'latent_dim': 512,  # latent dimension
    'latent_heads': 8,  # number of heads for latent self attention, 8
    'cross_dim_head': 64,  # number of dimensions per cross attention head
    'latent_dim_head': 64,  # number of dimensions per latent self attention head
    'num_classes': 1000,  # output number of classes
    'fourier_encode_data': True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
    'self_per_cross_attn': 2,  # number of self attention blocks per cross attention
    'weight_tie_layers': False, # whether to weight tie layers (optional, as indicated in the diagram)
}

def select_model(arch, perceiver_params, **kwargs):

    if not perceiver_params:
        perceiver_params = DEFAULT_PERCEIVER_PARAMS

    if 'perceiverIO' in arch:
        # model = PerceiverIO(
        #     dim=32,  # dimension of sequence to be encoded
        #     queries_dim=32,  # dimension of decoder queries
        #     logits_dim=100,  # dimension of final logits
        #     depth=6,  # depth of net
        #     num_latents=256,
        #     # number of latents, or induced set points, or centroids. different papers giving it different names
        #     latent_dim=512,  # latent dimension
        #     cross_heads=1,  # number of heads for cross attention. paper said 1
        #     latent_heads=8,  # number of heads for latent self attention, 8
        #     cross_dim_head=64,  # number of dimensions per cross attention head
        #     latent_dim_head=64,  # number of dimensions per latent self attention head
        #     weight_tie_layers=False  # whether to weight tie layers (optional, as indicated in the diagram)
        # )
        #
        # seq = torch.randn(1, 512, 32)
        # queries = torch.randn(128, 32)
        #
        # logits = model(seq, queries=queries)  # (1, 128, 100) - (batch, decoder seq, logits dim)
        # return model
        raise NotImplemented(f'Architecture {arch} has not been found.')

    elif 'perceiver' in arch:
        model = Perceiver(
            input_channels=3,  # number of channels for each token of the input
            input_axis=2,  # number of axis for input data (2 for images, 3 for video)
            num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
            max_freq=10.,  # maximum frequency, hyperparameter depending on how fine the data is
            depth=perceiver_params['depth'],  # depth of net. The shape of the final attention mechanism will be:
            #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=perceiver_params['num_latents'],
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=perceiver_params['latent_dim'],  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=perceiver_params['latent_heads'],  # number of heads for latent self attention, 8
            cross_dim_head=perceiver_params['cross_dim_head'],  # number of dimensions per cross attention head
            latent_dim_head=perceiver_params['latent_dim_head'],  # number of dimensions per latent self attention head
            num_classes=perceiver_params['num_classes'],  # output number of classes
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=perceiver_params['weight_tie_layers'],  # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data=perceiver_params['fourier_encode_data'],
            # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn=perceiver_params['self_per_cross_attn']  # number of self attention blocks per cross attention
        )
        return model
    else:
        raise NotImplemented(f'Architecture {arch} has not been found.')


"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim):
        super(Network, self).__init__()

        self.arch = arch
        self.embed_dim = embed_dim
        self.name = self.arch

        self.perceiver_params = {
            'depth': 4,  # depth of net. The shape of the final attention mechanism will be: depth * (cross attention -> self_per_cross_attn * self attention)
            'num_latents': 256, # number of latents, or induced set points, or centroids. different papers giving it different names
            'latent_dim': 384,  # latent dimension
            'latent_heads': 6,  # number of heads for latent self attention, 8
            'cross_dim_head': 64,  # number of dimensions per cross attention head
            'latent_dim_head': 64,  # number of dimensions per latent self attention head
            'num_classes': embed_dim,  # output number of classes
            'fourier_encode_data': True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            'self_per_cross_attn': 2, # number of self attention blocks per cross attention
            'weight_tie_layers': False, # whether to weight tie layers (optional, as indicated in the diagram)
        }

        self.model = select_model(arch, self.perceiver_params)

        print(f'ARCHITECTURE:\ntype: {self.arch}\nembed_dims: {self.embed_dim}\n')


    def forward(self, x, **kwargs):

        # reshape input to perceiver arch [bs, 3, 224, 224] -> [bs, 224, 224, 3]
        x = x.permute(0, 2, 3, 1)

        z = self.model(x)
        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        return {'embeds': z, 'avg_features': None, 'features': None, 'extra_embeds': None}
