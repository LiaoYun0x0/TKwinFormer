import os
import time
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from tqdm import tqdm
import argparse
from src.models.matcher import Matcher
import cv2
from tqdm import tqdm
from glob import glob
from copy import deepcopy


def draw_match(img0,img1,qs,rs,mask=None,draw_skip=1):
    h0,w0,c0 = img0.shape
    h1,w1,c1 = img1.shape
    assert c0 == c1,'assert error'
    oimg = np.zeros((max(h0,h1),w0+w1,c0),np.uint8)
    oimg[:h0,:w0,:] = img0
    oimg[:h1,w0:w0+w1,:] = img1
    for i,(q,r) in enumerate(zip(qs,rs)):
        if i % draw_skip == 0:
            if mask is None or mask[i]:
                color = (0,255,0)
            else:
                color = (0,0,255)
            cv2.line(oimg,(int(q[0]),int(q[1])),(int(r[0]+w0),int(r[1])),color,1)
    return oimg 

def read_image(image_path,resize=640):
    image = cv2.imread(image_path)
    # return cv2.resize(image,(640,640))
    h,w = image.shape[:2]
    scale = resize / max(w,h)
    nw,nh = int(round(w*scale)),int(round(h*scale))
    image = cv2.resize(image,(nw,nh))
    oimg = np.zeros((resize,resize,3),np.uint8)
    oimg[:nh,:nw] = image
    return oimg

if __name__ == "__main__":
    model_config = {
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
    device = 'cuda:1'
    model = Matcher(model_config).to(device)
    model.eval()
    weight_path = 'weights/model_640000_0.1736.pth'
    ckpts = torch.load(weight_path,map_location=device)
    model.load_state_dict(ckpts['model'])
    thresh = 0.2
    img_size = 1152
    
    qpath = 'assets/12003890_f6c899bec0_o.jpg'
    rpath = 'assets/13866250_56e0509621_o.jpg'

    qimg = read_image(qpath,img_size)
    rimg = read_image(rpath,img_size)
    query = torch.from_numpy(cv2.cvtColor(qimg,cv2.COLOR_BGR2GRAY)).float()[None,None,...].to(device) / 255
    refer = torch.from_numpy(cv2.cvtColor(rimg,cv2.COLOR_BGR2GRAY)).float()[None,None,...].to(device) / 255
    with torch.no_grad():
        src_pts,dst_pts = model.forward_test(query,refer,thresh)
        print("num points:",src_pts.shape[0])
    src_pts = src_pts.cpu().numpy()
    dst_pts = dst_pts.cpu().numpy()
    
    mask = None
    if src_pts.shape[0] > 4:  
        _,mask = cv2.findHomography(src_pts,dst_pts,cv2.USAC_DEFAULT,5.0)
        print("num homo points",mask.sum())
    
    oimg = draw_match(qimg, rimg, src_pts, dst_pts, mask=mask)
    cv2.imwrite('assets/demo.jpg',oimg)
    