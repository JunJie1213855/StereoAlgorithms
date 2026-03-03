import torch.nn as nn
import torch.nn.functional as F
from .submodule import *


class Aggregation2D(nn.Module):
    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super(Aggregation2D, self).__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        conv0 = [ConvNeXtBlock(in_channels) for i in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1 = nn.Sequential(
            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2),
        )

        conv2_add = [ConvNeXtBlock(in_channels * 2) for i in range(blocks[1] - 1)]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = nn.Sequential(
            LayerNorm(in_channels * 2, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=2, stride=2),
        )

        conv4_add = [ConvNeXtBlock(in_channels * 4) for i in range(blocks[2] - 1)]
        self.conv4 = nn.Sequential(*conv4_add)

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False)
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False)
        )

        self.redir1 = ConvNeXtBlock(in_channels)
        self.redir2 = ConvNeXtBlock(in_channels * 2)

        if self.left_att:
            self.att4 = AttentionModule2D(in_channels, backbone_channels[0])
            self.att8 = AttentionModule2D(in_channels * 2, backbone_channels[1])
            self.att16 = AttentionModule2D(in_channels * 4, backbone_channels[2])

    def forward(self, x, features_left):
        x_4 = self.conv0(x)
        if self.left_att:
            x_4 = self.att4(x_4, features_left[0])

        x_8 = self.conv1(x_4)
        x_8 = self.conv2(x_8)
        if self.left_att:
            x_8 = self.att8(x_8, features_left[1])

        x_16 = self.conv3(x_8)
        x_16 = self.conv4(x_16)
        if self.left_att:
            x_16 = self.att16(x_16, features_left[2])

        x_8 = F.relu(self.upconv1(x_16) + self.redir2(x_8), inplace=True)
        x_4 = F.relu(self.upconv2(x_8) + self.redir1(x), inplace=True)

        return x_4


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        return x + input


class AttentionModule2D(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * cost