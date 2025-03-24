import torch
import torch.nn as nn
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

# 修改1：精简MLP结构 轻量化模型
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features//2  # Modified: 缩小扩展比例
        
        self.mlp1_lif = LIFNode(tau=tau_thr, detach_reset=True, 
                              surrogate_function=surrogate.Sigmoid(alpha=8.0), step_mode='m')
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = LIFNode(tau=tau_thr, detach_reset=True,
                              surrogate_function=surrogate.Sigmoid(alpha=8.0), step_mode='m')
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.mlp1_bn(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        
        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return self.mlp2_bn(x.flatten(0, 1)).reshape(T, B, C, H, W)

# 修改2：优化注意力模块
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

# 修改4：重构Tokenizer
class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, in_channels=2, embed_dims=128):  # Modified: 降低嵌入维度
        super().__init__()
        self.stem = nn.Sequential(
            LIFNode(tau=tau_thr, detach_reset=True, step_mode='m'),
            nn.Conv2d(in_channels, embed_dims//4, 3, stride=2, padding=1),  # 直接下采样
            nn.BatchNorm2d(embed_dims//4),
            nn.MaxPool2d(3, 2, 1)
        )
        self.stage1 = nn.Sequential(
            LIFNode(tau=tau_thr, detach_reset=True, step_mode='m'),
            nn.Conv2d(embed_dims//4, embed_dims//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims//2)
        )
        self.stage2 = nn.Sequential(
            LIFNode(tau=tau_thr, detach_reset=True, step_mode='m'),
            nn.Conv2d(embed_dims//2, embed_dims, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims)
        )

    def forward(self, x):
        x = self.stem(x.flatten(0,1)).reshape(*x.shape[:2], -1, 32, 32)
        x = self.stage1(x.flatten(0,1)).reshape(*x.shape[:2], -1, 16, 16)
        x = self.stage2(x.flatten(0,1)).reshape(*x.shape[:2], -1, 8, 8)
        return x, (None, None)

# 修改5：精简模型架构
class vit_snn(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, in_channels=2, num_classes=12,
                 embed_dims=128, num_heads=8, mlp_ratios=2, depths=4, 
                 pretrained_cfg=None, **kwargs):  
        super().__init__()
        self.depths = depths
        self.patch_embed = SpikingTokenizer(img_size_h, img_size_w, in_channels, embed_dims)
        
        block = nn.Sequential(*[
            SpikingTransformer(embed_dims, num_heads, mlp_ratios)
            for _ in range(depths)
        ])
        
        setattr(self, f"block", block)

        self.head = nn.Linear(embed_dims, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T,B,C,H,W]
        x, _ = self.patch_embed(x)
        x = self.block(x)
        return self.head(x.mean(dim=[0,3,4]))

@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(
        embed_dims=128, 
        num_heads=8, 
        mlp_ratios=2,
        depths=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model