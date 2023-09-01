"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout
import torch.nn as nn
from einops import rearrange
import math

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class TopKWindowAttention(nn.Module):
    def __init__(self,d_head,w=7,k=8,attention='linear'):
        super(TopKWindowAttention, self).__init__()
        self.w = w
        self.k = k
        self.d_head = d_head

        if attention == "linear":
            self.attention = LinearAttention()
        elif attention == "full":
            self.attention = FullAttention()
        else:
            raise NotImplementedError()

    def forward(self, q,k,v,q_mask=None,kv_mask=None):
        '''
        q = atten(
            q, 
            cat(topk_window_ks, kw_mean), 
            cat(topk_window_vs, vw_mean)
        )
        '''
        b,d,h,w = q.shape
        qw = rearrange(q, 'b d (m w1) (n w2) -> b (m n) (w1 w2) d',w1=self.w,w2=self.w)
        kw = rearrange(k, 'b d (m w1) (n w2) -> b (m n) (w1 w2) d',w1=self.w,w2=self.w)
        vw = rearrange(v, 'b d (m w1) (n w2) -> b (m n) (w1 w2) d',w1=self.w,w2=self.w)
        qw_mean = torch.mean(qw,dim=2)
        kw_mean = torch.mean(kw,dim=2)
        vw_mean = torch.mean(vw,dim=2)
        
        window_similarity = torch.einsum('bmd,bnd->bmn',qw_mean,kw_mean)
        topk_values,topk_indices = torch.topk(window_similarity,dim=-1,k=self.k) # [b, m, k]
        
        fine_keys = []
        fine_values = []
        for i in range(b):
            fine_keys.append(kw[i][topk_indices[i]])
            fine_values.append(vw[i][topk_indices[i]])
        
        m,n = h // self.w, w // self.w
        fine_keys = torch.stack(fine_keys).reshape(b,m*n,-1,d) # [B, m*n, k*w1*w2, D]
        fine_values = torch.stack(fine_values).reshape(b,m*n,-1,d)
        
        keys = torch.cat([fine_keys,torch.tile(kw_mean.unsqueeze(1),(1,m*n,1,1))],2)
        values = torch.cat([fine_values,torch.tile(vw_mean.unsqueeze(1),(1,m*n,1,1))],2)
        
        queries = rearrange(qw,'b nw ws (h d) -> (b nw) ws h d',d=self.d_head)
        keys = rearrange(keys,'b nw ws (h d) -> (b nw) ws h d',d=self.d_head)
        values = rearrange(values,'b nw ws (h d) -> (b nw) ws h d',d=self.d_head)

        message = self.attention(queries, keys, values, q_mask=None, kv_mask=None)  # [N, L, (H, D)]
        message = rearrange(message, '(b m n) (w1 w2) h d -> b (h d) (m w1) (n w2)',m=m,n=n,w1=self.w)
        return message

    
class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()
    
class ChannelAttention(nn.Module):
    def __init__(self,d_head=128):
        super().__init__()
        self.d_head=d_head
    def forward(self,q,k,v):
        '''
        q,k,v: [N,L,H,D]
        '''
        n,l,h,d = q.shape
        q = rearrange(q,'n (h1 d1) h0 d0 -> n h1 (h0 d0) d1',d1=self.d_head)
        k = rearrange(k,'n (h1 d1) h0 d0 -> n h1 (h0 d0) d1',d1=self.d_head)
        v = rearrange(v,'n (h1 d1) h0 d0 -> n h1 (h0 d0) d1',d1=self.d_head)
        attention = q @ k.transpose(-1,-2) / q.size(3) ** 0.5
        # attention = q @ k.transpose(-1,-2) / 10000
        attention = torch.softmax(attention,dim=-1)
        x = attention @ v
        x = rearrange(x, 'n h1 (h0 d0) d1 -> n (h1 d1) h0 d0',h0=h)
        return x
        
class TopkSpatialChannelAttention(nn.Module):
    def __init__(self,d_head=32,d_channel=128,w=8,k=8,spatial_attn_type='linear'):
        super().__init__()
        self.d_head = d_head
        self.d_channel = d_channel
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.spatial_attn = TopKWindowAttention(d_head,w,k,spatial_attn_type)
        
    def forward(self,q,k,v_spatial,v_channel,q_mask,kv_mask):
        m1 = self.spatial_attn(q,k,v_spatial,q_mask,kv_mask)
        m2 = self.channel_attention(q,k,v_channel)
        return self.alpha * m1 + (1-self.alpha) * m2 
    
    def channel_attention(self,q,k,v):
        '''
        q,k,v: [N,C,H,W]
        '''
        n,c,h,w = q.shape
        q,k,v = map(lambda x:x.view(n,c,-1,self.d_channel).transpose(1,2),[q,k,v])
        attention = q @ k.transpose(-1,-2) / q.size(3)**0.5
        attention = torch.softmax(attention,dim=-1)
        x = attention @ v
        x = x.transpose(1,2).reshape(n,c,h,w)
        return x
    
class SpatialChannelAttention(nn.Module):
    def __init__(self,d_channel=128,spatial_attn_type='linear'):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        if spatial_attn_type == 'linear':
            self.spatial_attn = LinearAttention()
        else:
            self.spatial_attn = FullAttention()
        self.channel_attn = ChannelAttention(d_head=d_channel)
    def forward(self,q,k,v_spatial,v_channel,q_mask,kv_mask):
        m1 = self.spatial_attn(q,k,v_spatial,q_mask,kv_mask)
        m2 = self.channel_attn(q,k,v_channel)
        return self.alpha * m1 + (1-self.alpha) * m2
    