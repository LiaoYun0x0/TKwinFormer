import os
import time
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from src.models.matcher import Matcher
import cv2
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from src.megadepth.data import build_test_dataset

cfg = {}
cfg["device"] = "cuda:1" if torch.cuda.is_available() else "cpu:0"
cfg['img_resize'] = 1152
cfg['batch_size'] = 1
cfg['thresh'] = 0.2

cfg['model'] = {
        'dim_conv_stem' : 64,
        "dims" : [128,192,256],
        "depths" : [0,1,2],
        "dropout" : 0.1,
        "d_spatial" : 32,
        "d_channel" : 128,
        "mbconv_expansion_rate":[1,1,1,2],
        "attn_depth": 8,
        "w":[1,1,72,72,36,36,18,18],
        "k":[1,1,2,2,4,4,8,8]
    }

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    # ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    ransac_thr = thresh / np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])
    # compute pose with cv2
    E, mask = cv2.findEssentialMat(kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)
    return aucs

def corner_error(h,w,H_pred,H_gt):
    corner = np.array([
        [0,0,1],
        [w-1,0,1],
        [w-1,h-1,1],
        [0,h-1,1]
    ],np.float32)
    warp_corner_gt = corner @ H_gt.T
    warp_corner_gt = warp_corner_gt[:,:2] / warp_corner_gt[:,2:]
    warp_corner_pred = corner @ H_pred.T
    warp_corner_pred = warp_corner_pred[:,:2] / warp_corner_pred[:,2:]
    error = np.mean(np.sqrt(np.sum(np.power(warp_corner_gt - warp_corner_pred,2),axis=-1)))
    return error
    
def compute_acc(model,dataloader,device,px=[3,5,10,20,30],homo_filter=True):
    acc = [0 for _ in range(len(px))]
    errors = []
    for data in dataloader:
        query = data['query'].to(device).float()
        refer = data['refer'].to(device).float()
        mm = np.float32(data['map_matrix'])[0]
        with torch.no_grad():
            preds = model(query,refer,None)
        pts0 = preds['mkpts0'][:,1:].cpu().numpy()
        pts1 = preds['mkpts1'].cpu().numpy()
        if homo_filter and pts0.shape[0] > 4:
            H,mask = cv2.findHomography(pts0,pts1,cv2.RANSAC,ransacReprojThreshold=16)
            pts0 = pts0[mask.squeeze()]
            pts1 = pts1[mask.squeeze()]
        pts0_map = np.concatenate([pts0,np.ones((pts0.shape[0],1))],axis= -1) @ mm.T
        pts0_map = pts0_map[:,:2] / pts0_map[:,2:]
        error = np.sqrt(np.sum(np.power(pts0_map - pts1,2),-1))
        # errors.append(np.mean(error))
        if pts0.shape[0] > 4:
            errors.append(corner_error(query.shape[0], query.shape[1],H, mm))
        else:
            errors.append(10000)
        for i in range(len(px)):
            acc[i] += np.mean(error < px[i])
    for i in range(len(px)):
        acc[i] /= len(dataloader)
    msg_acc = ["acc@{}px: {:.4f}\n".format(px[i],acc[i]) for i in range(len(px))]
    auc = error_auc(errors, px)
    msg_auc = ["auc@{}px: {:.4f}\n".format(px[i],auc[i]) for i in range(len(px))]
    msg = "".join(msg_acc+msg_auc)
    print(msg)

def totensor(data,device):
    transfer_list = ['image0','image1','depth0','depth1','T_0to1','T_1to0','K0','K1','scale0','scale1']
    for k,v in data.items():
        if k in transfer_list:
            data[k] = data[k].to(device)
   
@torch.no_grad()
def eval(cfg):
    model = Matcher(cfg['model']).to(cfg['device'])
    model.eval()
    weight_path = 'weights/model_640000_0.1736.pth'
    ckpts = torch.load(weight_path,map_location=cfg['device'])
    model.load_state_dict(ckpts['model'])
    
    root_dir = '/first_disk/MegaDepth_v1'
    npz_root = 'assets/megadepth_test_1500_scene_info'
    test_list = 'assets/megadepth_test_1500_scene_info/megadepth_test_1500.txt'
    dataset_test = build_test_dataset(root_dir, npz_root, test_list, 'val', cfg['img_resize'],score=0.0,max_data_size=1e5)
    eval_loaders =  DataLoader(
        dataset_test,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4
    )
    print(len(eval_loaders))

    R_errors, t_errors,inliers = [],[],[]
    tbar = tqdm(eval_loaders)
    for i,data in enumerate(tbar):
        totensor(data, cfg['device'])
        with torch.no_grad():
            src_pts,dst_pts = model.forward_test(data['image0'],data['image1'],cfg['thresh'])
        src_pts = (src_pts * data['scale0'][0]).cpu().numpy()
        dst_pts = (dst_pts * data['scale1'][0]).cpu().numpy()
        
        ret = estimate_pose(src_pts, dst_pts, data['K0'][0].cpu().numpy(), data['K1'][0].cpu().numpy(),thresh=0.5)
        if ret is None:
            R_errors.append(np.inf)
            t_errors.append(np.inf)
        else:
            R, t, _inliers = ret
            t_err, R_err = relative_pose_error(data['T_0to1'][0].cpu().numpy(), R, t, ignore_gt_t_thr=0.0)
            R_errors.append(R_err)
            t_errors.append(t_err)
    
    R_errors = np.array(R_errors,dtype=np.float32)
    t_errors = np.array(t_errors,dtype=np.float32)
    thresholds = [5, 10, 20, 30, 40, 50, 60, 70]
    rauc = error_auc(R_errors, thresholds)
    tauc = error_auc(t_errors, thresholds)
    print(rauc)
    print(tauc)
    pose_errors = np.maximum(R_errors,t_errors)
    aucs = error_auc(pose_errors, thresholds)
    print(aucs)
        
if __name__ == "__main__":
    eval(cfg)
