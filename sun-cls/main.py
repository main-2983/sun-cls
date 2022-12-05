import torch
import torch.nn as nn


feats = torch.rand((1, 10, 48, 48))
rate = 64
conv = nn.Conv2d(
    in_channels=10,
    out_channels=10,
    kernel_size=8,
    stride=8
)
linear = nn.Linear(
    in_features=10 * rate,
    out_features=10
)
flatten_feat = feats.view(1, 10, 48 * 48).transpose(-1, -2)
B, L, C = flatten_feat.shape
reshape_feat = feats.view(1, L // rate, C * rate)

conv_weight_shape = conv.weight.shape
conv_new_weight = nn.Parameter(torch.ones(conv_weight_shape))
linear_weight_shape = linear.weight.shape
linear_new_weight = nn.Parameter(torch.ones(linear_weight_shape))

conv.weight = conv_new_weight
linear.weight = linear_new_weight

conv_out = conv(feats)
linear_out = linear(reshape_feat)

conv_out = conv_out.view(1, 10, conv_out.shape[-1] * conv_out.shape[-2]).transpose(-1, -2)
print(conv_out - linear_out)