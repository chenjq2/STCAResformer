import torch
import torch.nn as nn
import Alayers
from spikingjelly.activation_based.neuron import ParametricLIFNode, LIFNode
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from timm.models import create_model

__all__ = ['Spikingformer']
tau_thr = 2.0

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features//2

        self.mlp1_lif = LIFNode(tau=tau_thr, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid(alpha=8.0, spiking=True), step_mode='m')
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = LIFNode(tau=tau_thr, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid(alpha=8.0, spiking=True), step_mode='m')
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.mlp1_bn(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        
        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return self.mlp2_bn(x.flatten(0, 1)).reshape(T, B, C, H, W)

class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):  # 确保num_heads是dim的因数
        super().__init__()
        assert dim % num_heads == 0, "dim必须能被num_heads整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 保持proj_lif不变
        self.proj_lif = LIFNode(tau=tau_thr, detach_reset=True,
                               surrogate_function=surrogate.Sigmoid(alpha=8.0), step_mode='m')
        
        # 修改QKV生成方式
        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=qkv_bias)  # 输出通道改为3*dim
        self.qkv_bn = nn.BatchNorm1d(dim * 3)
        
        # 投影层保持不变
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape  # [10,16,128,8,8]
        N = H * W  # 64
        
        # 保持LIF节点
        x = self.proj_lif(x)  # 形状保持[T,B,C,H,W]
        
        # 生成QKV
        x_flat = x.flatten(0, 1)  # [T*B, C, H, W]
        x_flat = x_flat.view(x_flat.size(0), C, N)  # 修正1：展平空间维度 [160,128,64]
        qkv = self.qkv_bn(self.qkv_conv(x_flat))  # [160, 384, 64]
        
        # 拆分QKV
        q, k, v = qkv.chunk(3, dim=1)  # 每个[160,128,64]
        
        # 多头处理
        q = q.view(T, B, self.num_heads, self.head_dim, N)
        k = k.view(T, B, self.num_heads, self.head_dim, N)
        v = v.view(T, B, self.num_heads, self.head_dim, N)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        x = (attn @ v)  # [10,16,8,16,64]
        
        # 合并多头
        x = x.transpose(2, 3).reshape(T, B, C, H, W)  # [10,16,128,8,8]
        
        # 修正2：正确的投影维度处理
        x_flat = x.flatten(0, 1).view(-1, C, N)  # [160,128,64]
        x = self.proj_bn(self.proj_conv(x_flat))  # [160,128,64]
        x = x.view(T, B, C, H, W)  # 恢复原始形状
        
        return x

# 修改3：简化Transformer块
class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, drop=0.):  # Modified: mlp_ratio缩小
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim*mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x.permute(0,1,3,4,2)).permute(0,1,4,2,3))
        x = x + self.mlp(self.norm2(x.permute(0,1,3,4,2)).permute(0,1,4,2,3))
        return x


class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=260, img_size_w=346, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        # self.proj_lif = ParametricLIFNode(init_tau=tau_thr, detach_reset=True, backend='torch', surrogate_function=surrogate.ATan(alpha=8.0, spiking=True), step_mode='m')
        # self.proj_lif = LIFNode(tau=tau_thr, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid(alpha=8.0, spiking=True), step_mode='m')
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.TCJA1 = Alayers.TCJA(4, 4, 10, embed_dims//8)
        # self.proj1_lif = ParametricLIFNode(init_tau=tau_thr, detach_reset=True, backend='torch', surrogate_function=surrogate.ATan(alpha=8.0, spiking=True), step_mode='m')
        self.proj1_lif = LIFNode(tau=tau_thr, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid(alpha=8.0, spiking=True), step_mode='m')
        self.proj1_conv = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims//4)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.TCJA2 = Alayers.TCJA(4, 4, 10, embed_dims//4)
        # self.proj2_lif = ParametricLIFNode(init_tau=tau_thr, detach_reset=True, backend='torch', surrogate_function=surrogate.ATan(alpha=8.0, spiking=True), step_mode='m')
        self.proj2_lif = LIFNode(tau=tau_thr, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid(alpha=8.0, spiking=True), step_mode='m')
        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims//2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.TCJA3 = Alayers.TCJA(2, 4, 10, embed_dims//2)
        # self.proj3_lif = ParametricLIFNode(init_tau=tau_thr, detach_reset=True, backend='torch', surrogate_function=surrogate.ATan(alpha=8.0, spiking=True), step_mode='m')
        self.proj3_lif = LIFNode(tau=tau_thr, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid(alpha=8.0, spiking=True), step_mode='m')
        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.TCJA4 = Alayers.TCJA(4, 4, 10, embed_dims)
        # self.proj4_lif = ParametricLIFNode(init_tau=tau_thr, detach_reset=True, backend='torch', surrogate_function=surrogate.ATan(alpha=8.0, spiking=True), step_mode='m')
        self.proj4_lif = LIFNode(tau=tau_thr, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid(alpha=8.0, spiking=True), step_mode='m')
        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # x = self.proj_lif(x)
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x)
        x = self.maxpool(x)
        H, W = x.shape[2], x.shape[3]
        x = x.reshape(T, B, -1, H, W)

        # x = self.TCJA1(x).contiguous()
        x = self.proj1_lif(x).flatten(0, 1)
        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.maxpool1(x)
        H, W = x.shape[2], x.shape[3]
        x = x.reshape(T, B, -1, H, W)

        # x = self.TCJA2(x).contiguous()
        x = self.proj2_lif(x).flatten(0, 1)
        x = self.proj2_conv(x)
        x = self.proj2_bn(x)
        x = self.maxpool2(x)
        H, W = x.shape[2], x.shape[3]
        x = x.reshape(T, B, -1, H, W)

        # x = self.TCJA3(x).contiguous()
        x = self.proj3_lif(x).flatten(0, 1)
        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.maxpool3(x)
        H, W = x.shape[2], x.shape[3]
        x = x.reshape(T, B, -1, H, W)

        # x = self.TCJA4(x).contiguous()
        x = self.proj4_lif(x).flatten(0, 1)
        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        H, W = x.shape[2], x.shape[3]
        x = x.reshape(T, B, -1, H, W)

        return x, (H, W)

class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=260, img_size_w=346, patch_size=16, in_channels=2, num_classes=10,
                 embed_dims=128, num_heads=8, mlp_ratios=2, qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=2, sr_ratios=1, T=16, pretrained_cfg=None, pretrained_cfg_overlay=None,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        # print("depths is", depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SpikingTokenizer(img_size_h=img_size_h,
                          img_size_w=img_size_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        block = nn.ModuleList([SpikingTransformer(embed_dims, num_heads, mlp_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(
        patch_size=16, embed_dims=128, num_heads=16, mlp_ratios=2,
        in_channels=2, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=2, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model















