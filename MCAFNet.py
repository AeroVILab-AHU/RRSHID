import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops.layers.torch import Rearrange
from einops import rearrange
from mmengine.model import BaseModule
import typing as t
import math
from torchvision.ops import DeformConv2d




class MFIBA(nn.Module):
    def __init__(self, dim, x=8, y=8, bias=False):
        super(MFIBA, self).__init__()

        partial_dim = int(dim // 4)
        # 动态权重生成器
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 4, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.hw = nn.Parameter(torch.ones(1, partial_dim, x, y), requires_grad=True)
        self.conv_hw = nn.Conv2d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.ch = nn.Parameter(torch.ones(1, 1, partial_dim, x), requires_grad=True)
        self.conv_ch = nn.Conv1d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.cw = nn.Parameter(torch.ones(1, 1, partial_dim, y), requires_grad=True)
        self.conv_cw = nn.Conv1d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.conv_4 = nn.Conv2d(partial_dim, partial_dim, kernel_size=1, bias=bias)

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        # 使用 Swish 激活函数代替 ReLU
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )


    def forward(self, x):
        input_ = x
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # 动态权重计算
        weights = self.weight_generator(x)  # [B,4,1,1]
        w1, w2, w3, w4 = torch.chunk(weights, 4, dim=1)
        # HW分支
        hw_scale = F.interpolate(self.hw, size=x1.shape[2:4], mode='bilinear')
        x1 = x1 * w1 * self.conv_hw(hw_scale)

        # CH分支
        x2 = x2.permute(0, 3, 1, 2)
        ch_scale = F.interpolate(self.ch, size=x2.shape[2:4], mode='bilinear').squeeze(0)
        x2 = x2 * w2 * self.conv_ch(ch_scale).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)

        # CW分支
        x3 = x3.permute(0, 2, 1, 3)
        cw_scale = F.interpolate(self.cw, size=x3.shape[2:4], mode='bilinear').squeeze(0)
        x3 = x3 * w3 * self.conv_cw(cw_scale).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)

        # 点卷积分支
        x4 = x4 * w4 * self.conv_4(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        x = self.mlp(x) + input_
        return x


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2



class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super(BasicLayer, self).__init__()

        self.blocks = nn.ModuleList([MFIBA(dim, dim) for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class MFAFM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFAFM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)


        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=4,
            batch_first=False
        )


        self.fusion_conv = nn.Conv2d(in_channels, out_channels , kernel_size=1)


        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, skip1, skip2):

        skip1 = self._adjust_feature(skip1, x.shape)
        skip2 = self._adjust_feature(skip2, x.shape)


        x1 = self.conv1(x)
        x2 = self.conv2(skip1)
        x3 = self.conv3(skip2)


        B, C, H, W = x1.shape
        x1_flat = x1.view(B, C, -1).permute(2, 0, 1)  # [HW, B, C]
        x2_flat = x2.view(B, C, -1).permute(2, 0, 1)  # [HW, B, C]
        x3_flat = x3.view(B, C, -1).permute(2, 0, 1)  # [HW, B, C]

        attn_out, _ = self.cross_attn(x1_flat, x2_flat, x3_flat)
        attn_out = attn_out.permute(1, 2, 0).view(B, C, H, W)


        out = self.fusion_conv(attn_out)
        residual = self.residual_conv(x)
        return out + residual

    def _adjust_feature(self, feature, target_shape):
        if feature.shape != target_shape:
            feature = F.interpolate(feature, size=target_shape[2:], mode='bilinear', align_corners=False)
        if feature.shape[1] != self.in_channels:
            feature = F.adaptive_avg_pool3d(feature, (self.in_channels, feature.shape[2], feature.shape[3]))
        return feature


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
#


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x



class CSAM(nn.Module):
    def __init__(self, dim, up_scale=2, bias=False):
        super(CSAM, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)


        self.up = nn.PixelShuffle(up_scale)


        self.qk_pre = nn.Conv2d(int(dim // (up_scale ** 2)), 3, kernel_size=to_2tuple(1), bias=bias)
        self.qk_post = nn.Sequential(LayerNorm2d(3),
                                     nn.Conv2d(3, int(dim * 2), kernel_size=to_2tuple(1), bias=bias))
        self.v = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias)
        )


        self.conv = nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias)


        self.norm = LayerNorm2d(dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias)
        )


        self.color_correction = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        nn.init.eye_(self.color_correction.weight.squeeze().data)

        self.proj_head = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qk = self.qk_pre(self.up(x))
        fake_image = self.color_correction(qk)


        anchor = self.proj_head(F.normalize(fake_image, dim=1))

        qk = self.qk_post(qk).reshape(b, 2, c, -1).transpose(0, 1)
        q, k = qk[0], qk[1]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        v = self.v(x)
        v_ = v.reshape(b, c, h * w)
        attn = (q @ k.transpose(-1, -2)) * self.alpha
        attn = attn.softmax(dim=-1)
        x = (attn @ v_).reshape(b, c, h, w) + self.conv(v)
        x = self.norm(x)
        x = self.proj(x)
        return x, fake_image, anchor



class MCAFNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[8, 8, 16, 8, 8]):
        super(MCAFNet, self).__init__()

        # setting
        self.patch_size = 4

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        # self.skip1 = MFAFM(in_channels=embed_dims[0], out_channels=embed_dims[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        # self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        # self.skip2 = MFAFM(in_channels=embed_dims[1], out_channels=embed_dims[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        # self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])


        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]

        self.csam1 = CSAM(embed_dims[3], up_scale=2)

        self.fusion1 = MFAFM(embed_dims[3], embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]

        self.csam2 = CSAM(embed_dims[4], up_scale=1)


        self.fusion2 = MFAFM(embed_dims[4], embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)



    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x


        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)

        x, fake_image_x4, anchor_x4= self.csam1(x)
        x = self.fusion1(x, skip1, skip2) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x, fake_image_x2, anchor_x2 = self.csam2(x)
        x = self.fusion2(x, skip1, skip2) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x, fake_image_x4, fake_image_x2, anchor_x4, anchor_x2


    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        feat, fake_image_x4, fake_image_x2, anchor_x4, anchor_x2= self.forward_features(x)

        K, B = torch.split(feat, [1, 3], dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]

        return x, fake_image_x4, fake_image_x2, anchor_x4, anchor_x2



def MCAFNet_l():
    return MCAFNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[8, 8, 16, 8, 8])


if __name__ == '__main__':
    block = MCAFNet(24).to('cuda')
    input = torch.randn(1, 24, 32, 16).to('cuda')
    out = block(input)
    print(input.size())
    print(out.size())
