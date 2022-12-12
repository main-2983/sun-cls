import random

import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmcv.cnn import build_activation_layer

from timm.models.layers import trunc_normal_


class FFN(BaseModule):
    """
    Implementation of FFN using 1*1 convolution OpenMMLab style
    """
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 act_cfg=dict(
                     type='GELU'
                 ),
                 drop_rate=0.,
                 init_cfg=None):
        super(FFN, self).__init__(init_cfg)

        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.fc1 = nn.Conv2d(in_channels,
                             hidden_channels,
                             kernel_size=1)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_channels,
                             out_channels,
                             kernel_size=1)
        self.drop = nn.Dropout(drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)

        return out + identity


class GhostFFN(BaseModule):
    """
    Implementation of GhostFFN using 1*1 convolution OpenMMLab style
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 ghost_kernel=3,
                 act_cfg=dict(
                     type='GELU'
                 ),
                 drop_rate=0.,
                 init_cfg=None):
        super(GhostFFN, self).__init__(init_cfg)

        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        ghost_channels = hidden_channels - in_channels
        self.ghost_fc = nn.Conv2d(in_channels,
                                  ghost_channels,
                                  kernel_size=ghost_kernel,
                                  padding=ghost_kernel//2,
                                  groups=in_channels)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_channels,
                             out_channels,
                             kernel_size=1)
        self.drop = nn.Dropout(drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        out = self.ghost_fc(x)
        out = torch.cat([out, x], dim=1)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)

        return out + identity
