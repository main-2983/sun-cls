import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_

from mmcv.runner import BaseModule

from .linear import GhostFFN
from .patchembed import PatchEmbed3D


# specify this in order to use model factory in timm
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'ghostpool_s': _cfg(crop_pct=0.9),
    'ghostpool_m': _cfg(crop_pct=0.95),
}


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    Args:
        pool_size: kernel size of Average Pooling module
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class ModifiedLayerNorm(nn.GroupNorm):
    """
    Modified Layer Norm in https://arxiv.org/abs/2111.11418
    - LayerNorm which used in Transformer computes mean and variance on dimension C,
    its weight is of shape (C,)
    - nn.LayerNorm on (B, C, H, W) computes mean and variance on dimension (C, H, W) -> correct - is:
    >>> nn.LayerNorm(normalized_shape=(C, H, W))
    which has weight of shape (C, H, W) -> This is not correct, we need weights to have shape (C,)
    - Modified Layer Norm computes mean and variance on dimension (C, H, W) -> correct
    which has weight of shape (C,) -> correct
    For more explanation, refer to https://github.com/sail-sg/poolformer/issues/9
    """
    def __init__(self,
                 num_channels,
                 **kwargs):
        super(ModifiedLayerNorm, self).__init__(1, num_channels, **kwargs)


class GhostPoolBlock(BaseModule):
    """
    Implementation of PoolFormer Block with FFN replaced with GhostFFN
    A GhostPool Block consist of:
    - A Norm layer at the start
    - Token Mixer
    - Norm layer
    - Channel MLP
    Args:
        embed_dims: num embedded channels
        pool_size: kernel size of Pooling module
        mlp_ratio: expand ratio of FFN
        act_cfg: cfg for activation function
        norm_layer: Norm layer
        drop_rate: rate for nn.DropOut
        drop_path: rate for DropPath (Stochastic Depth)
        use_layer_scale: refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self,
                 embed_dims,
                 pool_size=3,
                 mlp_ratio=4.0,
                 act_cfg=dict(
                     type='GELU'
                 ),
                 norm_layer=ModifiedLayerNorm,
                 drop_rate=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 layer_scale_init_val=1e-5):
        super(GhostPoolBlock, self).__init__()
        self.norm1 = norm_layer(embed_dims)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(embed_dims)
        mlp_hidden_chans = int(embed_dims * mlp_ratio)
        self.mlp = GhostFFN(in_channels=embed_dims,
                            hidden_channels=mlp_hidden_chans,
                            out_channels=embed_dims,
                            act_cfg=act_cfg,
                            drop_rate=drop_rate)
        # The following two techniques are useful to train deep PoolFormers
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale1 = nn.Parameter(
                layer_scale_init_val * torch.ones((embed_dims)), requires_grad=True
            )
            self.layer_scale2 = nn.Parameter(
                layer_scale_init_val * torch.ones((embed_dims)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GhostPool(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 320, 512],
                 num_stages=4,
                 num_layers=[2, 2, 6, 2],
                 patch_size=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 drop_rate=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_layer=ModifiedLayerNorm,
                 use_layer_scale=True,
                 layer_scale_init_val=1e-5,
                 init_cfg=None):
        super(GhostPool, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.strides    = strides

        assert num_stages == len(num_layers) == len(patch_size) \
            == len(strides)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        self.stages = nn.ModuleList()
        for i, stage_i in range(self.num_stages):
            patch_embed = PatchEmbed3D(
                in_channels=in_channels,
                out_channels=embed_dims[i],
                patch_size=patch_size[i],
                stride=strides[i],
                padding=patch_size[i] // 2,
                norm_cfg=None
            )
            blocks = []
            for idx in range(num_layers[i]):
                drop_path_i = drop_path * (idx + sum(num_layers[:i])) / (sum(num_layers) - 1)
                blocks.append(
                    GhostPoolBlock(
                        embed_dims=embed_dims[i],
                        pool_size=3,
                        mlp_ratio=mlp_ratio,
                        act_cfg=act_cfg,
                        norm_layer=norm_layer,
                        drop_rate=drop_rate,
                        drop_path=drop_path_i,
                        use_layer_scale=use_layer_scale,
                        layer_scale_init_val=layer_scale_init_val
                ))
            blocks = nn.Sequential(*blocks)
            in_channels = embed_dims[i]
            self.stages.append(
                nn.ModuleList([patch_embed, blocks])
            )