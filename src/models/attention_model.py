import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from copy import deepcopy
from functools import partial
import torch.cuda.amp as amp
from kornia.utils import create_meshgrid
from src.models.attention import LinearAttention,FullAttention,SpatialChannelAttention, \
    TopKWindowAttention,TopkSpatialChannelAttention

class ConvBNGelu(nn.Module):
    def __init__(self,c_in,c_out,k,s):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in,c_out,k,s,k//2,bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU()
        )
    def forward(self,x):
        return self.net(x)
    
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
class MB(nn.Module):
    def __init__(self,dim_in,dim_out,mlp_ratio=4,prenorm=False,afternorm=True,k=3,stride=1):
        super().__init__()
        dim_mid = int(dim_in * mlp_ratio)
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim_in) if prenorm else nn.Identity(),
            nn.Conv2d(dim_in, dim_mid, 1,bias=False),
            nn.BatchNorm2d(dim_mid),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_mid, k,stride,k//2,groups=dim_mid,bias=False),
            nn.BatchNorm2d(dim_mid),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_out, 1,bias=False),
            nn.BatchNorm2d(dim_out) if afternorm else nn.Identity()
        )
    def forward(self,x):
        x = self.net(x)
        return x

class ResidualMB(nn.Module):
    def __init__(self,dim_in,dim_out,mlp_ratio=4,prenorm=False,afternorm=True,k=3,stride=1,dropout=0.):
        super().__init__()
        dim_mid = int(dim_in * mlp_ratio)
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim_in) if prenorm else nn.Identity(),
            nn.Conv2d(dim_in, dim_mid, 1,bias=False),
            nn.BatchNorm2d(dim_mid),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_mid, k,stride,1,groups=dim_mid,bias=False),
            nn.BatchNorm2d(dim_mid),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_out, 1,bias=False),
            nn.BatchNorm2d(dim_out) if afternorm else nn.Identity()
        )
        self.main = nn.Sequential(
            nn.MaxPool2d(k, stride, k//2) if stride > 1 else nn.Identity(),
            nn.Conv2d(dim_in, dim_out, 1, bias=False) if dim_in != dim_out else nn.Identity()
        )
        self.dropout = DropPath(dropout)
            
    def forward(self,x):
        return self.main(x) + self.dropout(self.net(x))
        
    
class ConvBlock(nn.Module):
    def __init__(self,dim,dropout=0.,mlp_ratio=4):
        super().__init__()
        self.conv = MB(dim,dim,mlp_ratio,False,True)
        self.mlp = MB(dim,dim,mlp_ratio,False,True)
        self.dropout = DropPath(dropout)
    def forward(self,x):
        x = x + self.dropout(self.conv(x))
        x = x + self.dropout(self.mlp(x))
        return x


class TopkAttentionLayer(nn.Module):
    def __init__(self,d_model,d_head,w=8,k=8,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.w = w
        self.k = k
        self.d_model = d_model
        self.d_head = d_head
        self.nhead = self.d_model // self.d_head
        
        self.pre_normact = nn.Sequential(
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.q_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)   
        self.k_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)  
        self.v_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)  
        
        if w == 1:
            if attention == 'linear':
                self.attention = LinearAttention()
            elif attention == 'full':
                self.attention = FullAttention()
            else:
                raise NotImplementedError()
        else:
            self.attention = TopKWindowAttention(d_head,w=w,k=k,attention=attention)
        self.merge = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
        )
        self.mlp = MB(d_model,d_model,mlp_ratio,False,True)
        self.dropout = DropPath(dropout)
        
    def forward(self, x0,x1=None,q_mask=None,kv_mask=None):
        b,d,h,w = x0.shape
        if x1 is None:
            x1 = x0
        _x0,_x1 = self.pre_normact(x0),self.pre_normact(x1)
        q = self.q_proj(_x0)
        k = self.k_proj(_x1)
        v = self.v_proj(_x1)
        
        if self.w == 1:
            q = rearrange(q, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
            k = rearrange(k, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
            v = rearrange(v, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
            message = self.attention(q, k, v, q_mask=q_mask, kv_mask=kv_mask)  # [N, L, (H, D)]
            message = rearrange(message,'b (h w) heads d -> b (heads d) h w',h=h)
        else:
            message = self.attention(q,k,v)
        x = x0 + self.dropout(self.merge(message))
        x = x + self.dropout(self.mlp(x))
        return x
    
class TopkAttentionBlock(nn.Module):
    def __init__(self,d_model,d_head,w=8,k=8,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.self_attn = TopkAttentionLayer(d_model, d_head,w,k,dropout,attention,mlp_ratio,)
        self.cross_attn = TopkAttentionLayer(d_model, d_head,w,k,dropout,attention,mlp_ratio)
    def forward(self,x):
        x = self.self_attn(x)
        x0,x1 = torch.chunk(x, 2,dim=0)
        x0,x1 = self.cross_attn(x0,x1),self.cross_attn(x1,x0)
        x = torch.concat([x0,x1],dim=0)
        return x

class AttentionLayer(nn.Module):
    def __init__(self,d_model,d_head,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.nhead = self.d_model // self.d_head
        
        self.pre_norm = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        if attention == 'linear':
            self.attention = LinearAttention()
        elif attention == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError()
        
        self.merge = nn.Sequential(
            nn.Linear(d_model, d_model),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*mlp_ratio,d_model),
            nn.Dropout(dropout)
        )
        self.dropout = DropPath(dropout)
        
    def forward(self, x0,x1=None,q_mask=None,kv_mask=None):
        '''
        x0,x1: [n, l, d]
        q_mask,kv_mask: [n, l]
        '''
        if x1 is None:
            x1 = x0
        _x0,_x1 = self.pre_norm(x0), self.pre_norm(x1)
        q = self.q_proj(_x0)
        k = self.k_proj(_x1)
        v = self.v_proj(_x1)
        
        q = rearrange(q, ' n l (h d) -> n l h d', h=self.nhead)
        k = rearrange(k, ' n s (h d) -> n s h d', h=self.nhead)
        v = rearrange(v, ' n s (h d) -> n s h d', h=self.nhead)
        message = self.attention(q, k, v, q_mask=None, kv_mask=None)
        message = rearrange(message,'n l h d -> n l (h d)')
        
        x = x0 + self.dropout(self.merge(message))
        x = x + self.dropout(self.mlp(x))
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self,d_model,d_head,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.self_attn = AttentionLayer(d_model, d_head,dropout,attention,mlp_ratio)
        self.cross_attn = AttentionLayer(d_model, d_head,dropout,attention,mlp_ratio)
        
    def forward(self, x0,x1):
        x0, x1 = self.self_attn(x0,x0), self.self_attn(x1,x1)
        x0, x1 = self.cross_attn(x0, x1), self.cross_attn(x1,x0)
        return x0, x1




    
class TopkSpatialChannelAttentionLayer(nn.Module):
    def __init__(self,d_model,d_spatial,d_channel,w=8,k=8,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_spatial
        self.d_channel = d_channel
        self.nhead = self.d_model // self.d_head
        self.w = w
        
        self.pre_normact = nn.Sequential(
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.q_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)   
        self.k_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)  
        self.v_proj_spatial = nn.Conv2d(d_model, d_model, 1,1,bias=False)
        self.v_proj_channel = nn.Conv2d(d_model, d_model, 1,1,bias=False)

        if self.w == 1:
            self.attention = SpatialChannelAttention(d_channel,attention)
        else:
            self.attention = TopkSpatialChannelAttention(d_spatial,d_channel,w,k,attention)
        
        self.merge = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
        )
        self.mlp = MB(d_model,d_model,mlp_ratio,False,True)
        self.dropout = DropPath(dropout)
        
    def forward(self, x0, x1=None,q_mask=None,kv_mask=None):
        '''
        x0,x1: [N, C, H, W]
        q_mask,kv_mask: [N, (H W)]
        '''
        b,d,h,w = x0.shape
        if x1 is None:
            x1 = x0
        _x0,_x1 = self.pre_normact(x0),self.pre_normact(x1)
        q = self.q_proj(_x0)
        k = self.k_proj(_x1)
        v_spatial = self.v_proj_spatial(_x1)
        v_channel = self.v_proj_channel(_x1)
        if self.w == 1:
            q = rearrange(q, 'n (heads d) h w -> n (h w) heads d',heads=self.nhead)
            k = rearrange(k, 'n (heads d) h w -> n (h w) heads d',heads=self.nhead)
            v_spatial = rearrange(v_spatial, 'n (heads d) h w -> n (h w) heads d',heads=self.nhead)
            v_channel = rearrange(v_channel, 'n (heads d) h w -> n (h w) heads d',heads=self.nhead)
            message = self.attention(q,k,v_spatial,v_channel,q_mask,kv_mask)
            message = rearrange(message,'n (h w) heads d -> n (heads d) h w',h=h)
        else:
            message = self.attention(q,k,v_spatial,v_channel,q_mask,kv_mask)
        
        x = x0 + self.dropout(self.merge(message))
        x = x + self.dropout(self.mlp(x))
        return x

class SelfTCCrossT(nn.Module):
    def __init__(self,d_model,d_spatial,d_channel,w=8,k=8,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.self_attn = TopkSpatialChannelAttentionLayer(d_model, d_spatial,d_channel,w,k,dropout,attention,mlp_ratio)
        self.cross_attn = TopkAttentionLayer(d_model, d_spatial,w,k,dropout,attention,mlp_ratio)
    def forward(self,x):
        x = self.self_attn(x)
        x0,x1 = torch.chunk(x, 2,dim=0)
        x0,x1 = self.cross_attn(x0,x1),self.cross_attn(x1,x0)
        x = torch.concat([x0,x1],dim=0)
        return x
    

class MBFormer_248_topk(nn.Module):
    def __init__(
        self,
        *,
        dim_conv_stem = 64,
        dims=[128,192,256],
        depths=[2,2,2],
        d_spatial = 32,
        d_channel = 128,
        mbconv_expansion_rate = [1,1,2,3],
        dropout = 0.1,
        in_chans = 1,
        attn_depth = 4,
        w=[8,8,8,8],
        k=[8,8,8,8],
    ):
        super().__init__()

        self.conv_stem = nn.Sequential(
            ConvBNGelu(in_chans, dim_conv_stem, 3, 2),
            ConvBNGelu(dim_conv_stem, dim_conv_stem, 3, 1),
        )

        self.num_stages = len(dims)

        self.d0 = nn.Sequential(
                ConvBNGelu(dim_conv_stem, dims[0], 3, 1),
                ConvBNGelu(dims[0], dims[0], 3, 1),
            ) if depths[0] == 0 else \
            nn.Sequential(
                *([ResidualMB(dim_conv_stem, dims[0],stride=1,mlp_ratio=1)] + \
                [ConvBlock(dims[0],mlp_ratio=mbconv_expansion_rate[0]) for _ in range(depths[0])])
            )
            
        self.d1 = nn.Sequential(
            *([ResidualMB(dims[0], dims[1],stride=2,mlp_ratio=1)] + \
            [ConvBlock(dims[1],mlp_ratio=mbconv_expansion_rate[1]) for _ in range(depths[1])])
        )
        self.d2 = nn.Sequential(
            *([ResidualMB(dims[1], dims[2],stride=2,mlp_ratio=1)] + \
            [ConvBlock(dims[2],mlp_ratio=mbconv_expansion_rate[2]) for _ in range(depths[2])])
        )
        
        self.u0 = nn.ModuleList([
            nn.Conv2d(dims[0], dims[1], 1,bias=False),
            nn.Sequential(
                ConvBNGelu(dims[1], dims[1], 3, 1),
                nn.Conv2d(dims[1], dims[0], 3,1,1,bias=False)
            )
        ])
        self.u1 = nn.ModuleList([
            nn.Conv2d(dims[1], dims[2], 1,bias=False),
            nn.Sequential(
                ConvBNGelu(dims[2], dims[2], 3, 1),
                nn.Conv2d(dims[2], dims[1], 3,1,1,bias=False)
            )
        ])
        self.attn = nn.Sequential(
            *[SelfTCCrossT(dims[-1], d_spatial, d_channel, w[i], k[i], dropout,'linear',mbconv_expansion_rate[3]) for i,_ in enumerate(range(attn_depth))]
        )
        
    def forward(self, x):
        outputs = []
        x = self.conv_stem(x)
        x0 = self.d0(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        for attn in self.attn:
            x2 = attn(x2)
        
        outputs.append(x2)
        x2_up = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=True)
        x1 = self.u1[1](self.u1[0](x1) + x2_up)
        outputs.append(x1)
        x1_up = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=True)
        x0 = self.u0[1](self.u0[0](x0) + x1_up)
        outputs.append(x0)
        return list(reversed(outputs))
