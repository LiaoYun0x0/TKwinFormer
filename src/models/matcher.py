import torch
import torch.nn as nn
import torchvision as tv
import random
import numpy as np
from einops import rearrange,repeat
from kornia.utils import create_meshgrid
import math
from sklearn.cluster import dbscan
from copy import deepcopy
from kornia.geometry.subpix import dsnt
from src.models.attention_model import MBFormer_248_topk,AttentionBlock
from src.models.utils import spvs_coarse

INF = 1e9
EPS = 1e-6

class Matcher(nn.Module):
    def __init__(self,config):
        super().__init__()
        dims = config['dims']
        self.backbone = MBFormer_248_topk(**config)
        self.fine_loftr = AttentionBlock(
            dims[0],
            d_head=config['d_spatial'],
            dropout=config['dropout'],
            mlp_ratio=config['mbconv_expansion_rate'][-1]            
        )
        self.down_proj = nn.Linear(dims[2], dims[0], bias=True)
        self.merge_feat = nn.Linear(2*dims[0], dims[0], bias=True)

    def compute_coarse_loss(self,conf,conf_gt,weight):
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = 1.0, 1.0
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.
        conf = torch.clamp(conf, EPS, 1-EPS)
        alpha = 0.25
        gamma = 2.0
        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
    
        loss = c_pos_w * loss_pos.mean()
        return loss
        
        
    def forward_train(self,data):
        features = self.backbone(torch.cat([data['image0'],data['image1']],dim=0))
        device=features[0].device
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })
        spvs_coarse(data, 8)
            
        feature_c = rearrange(features[2],'n c h w -> n (h w) c')
        feature_c0, feature_c1 = torch.chunk(feature_c, chunks=2,dim=0)
        feature_c0_norm, feature_c1_norm = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feature_c0, feature_c1])
        
        similarity_matrix = torch.einsum('nld,nsd->nls',feature_c0_norm, feature_c1_norm) / 0.1
        if 'mask0' in data:
            weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float().to(device)
            similarity_matrix.masked_fill_(~weight.bool(),-INF)
        else:
            weight = None
        cf = torch.softmax(similarity_matrix, dim=1) * torch.softmax(similarity_matrix, dim=2)
        loss_coarse = self.compute_coarse_loss(cf, data['conf_matrix_gt'], weight)
        
        # fine process
        b_ids, i_ids, j_ids = data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids']
        if len(b_ids) == 0:
            loss_fine = torch.tensor(0,device)
            return loss_coarse, loss_fine
        W=5
        WW = W**2
        feature_f_unfold = torch.nn.functional.unfold(features[0], kernel_size=(W,W), stride=W-1, padding=W//2)
        feature_f_unfold = rearrange(feature_f_unfold, 'n (c ww) l -> n l ww c',ww=WW)
        f0_unfold, f1_unfold = torch.chunk(feature_f_unfold,chunks=2,dim=0)
        f0_unfold = f0_unfold[b_ids, i_ids]  # [n, ww, cf]
        f1_unfold = f1_unfold[b_ids, j_ids]

        feat_c_win = self.down_proj(torch.cat([feature_c0[b_ids, i_ids],
                                                feature_c1[b_ids, j_ids]], 0))  # [2n, c]
        feat_cf_win = self.merge_feat(torch.cat([
            torch.cat([f0_unfold, f1_unfold], 0),  # [2n, ww, cf]
            repeat(feat_c_win, 'n c -> n ww c', ww=WW),  # [2n, ww, cf]
        ], -1))
        f0_unfold, f1_unfold = torch.chunk(feat_cf_win,chunks=2,dim=0)
        f0_unfold, f1_unfold = self.fine_loftr(f0_unfold,f1_unfold)
        
        
        feat_f0_picked = f0_unfold[:, WW//2, :]
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, f1_unfold)
        softmax_temp = 1. / f1_unfold.shape[-1]**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=EPS)), -1)  # [M]  clamp needed for numerical stability
        
        expec_f = coords_normalized
        scale = 2 * data['scale1'][b_ids] if 'scale1' in data else 2
        expec_f_gt = (data['spv_w_pt0_i'][b_ids, i_ids] - data['spv_pt1_i'][b_ids, j_ids]) / scale / (W//2)  # [M, 2]
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < 1.0
        # use std as weight that measures uncertainty
        inverse_std = 1. / torch.clamp(std, min=EPS)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            loss_fine = torch.tensor(0, device=device)
        else:
            # l2 loss with std
            offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
            loss_fine = (offset_l2 * weight[correct_mask]).mean()
        
        return loss_coarse, loss_fine
    
    def forward_test(self,query,refer, thresh=0.8):
        device = query.device
        b = query.shape[0]
        assert b == 1, 'only support batchsize 1'
        features = self.backbone(torch.cat([query,refer],dim=0))
        
        t = 3
        feature0,feature1 = features[t-1].split(b)
        
        b,d,h,w = feature0.shape
        feature0 = rearrange(feature0, 'b c h w -> b (h w) c')
        feature1 = rearrange(feature1, 'b c h w -> b (h w) c')
        
        # feature0, feature1 = map(lambda feat: feat / feat.shape[-1]**.5,
        #                        [feature0, feature1])
        # sm = torch.einsum('nld,nsd->nls',feature0,feature1) / 0.1
        feature0_norm, feature1_norm = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feature0, feature1])
        sm = torch.einsum('nld,nsd->nls',feature0_norm,feature1_norm) / 0.1
        
        cm = torch.softmax(sm,dim=1) * torch.softmax(sm,dim=2)
        # bindex,qindex,rindex = torch.where(cm>thresh)
        mask = cm > thresh
        mask = mask * (cm == torch.max(cm,dim=1,keepdim=True)[0]) * (cm == torch.max(cm, dim=2,keepdim=True)[0])
        bindex,qindex,rindex = torch.where(mask)
        if len(bindex) == 0:
            return torch.zeros((0,2),device=bindex.device),torch.zeros((0,2),device=bindex.device)
        qcoords = torch.stack([qindex % w, qindex // w],dim=-1)
        rcoords = torch.stack([rindex % w, rindex // w],dim=-1)
        
        
        W=5
        WW = W**2
        feature_f_unfold = torch.nn.functional.unfold(features[0], kernel_size=(W,W), stride=W-1, padding=W//2)
        feature_f_unfold = rearrange(feature_f_unfold, 'n (c ww) l -> n l ww c',ww=WW)
        f0_unfold, f1_unfold = torch.chunk(feature_f_unfold,chunks=2,dim=0)
        f0_unfold = f0_unfold[bindex, qindex]  # [n, ww, cf]
        f1_unfold = f1_unfold[bindex, rindex]

        feat_c_win = self.down_proj(torch.cat([feature0[bindex, qindex],
                                                feature1[bindex, rindex]], 0))  # [2n, c]
        feat_cf_win = self.merge_feat(torch.cat([
            torch.cat([f0_unfold, f1_unfold], 0),  # [2n, ww, cf]
            repeat(feat_c_win, 'n c -> n ww c', ww=WW),  # [2n, ww, cf]
        ], -1))
        f0_unfold, f1_unfold = torch.chunk(feat_cf_win,chunks=2,dim=0)
        f0_unfold, f1_unfold = self.fine_loftr(f0_unfold,f1_unfold)
        
        
        feat_f0_picked = f0_unfold[:, WW//2, :]
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, f1_unfold)
        softmax_temp = 1. / f1_unfold.shape[-1]**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, device).reshape(1, -1, 2)  # [1, WW, 2]

        qcoords = qcoords * 4
        rcoords = rcoords * 4 + coords_normalized * 2
        
        qcoords = qcoords * 2
        rcoords = rcoords * 2
        return qcoords,rcoords

        
        
            
            