import smtpd
import sys

sys.path.append("../")
import pdb
import time
import shutil
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
import random
import os
import shutil
import numpy as np
import cv2
import scipy.misc
import math
import matplotlib.pyplot as plt
import time
import trimesh
import nvdiffrast.torch as dr
from configs.parameter import Params
from dataset.patch import PatchDataset

if __name__ == "__main__":
    params = Params()
    dataset = PatchDataset(params=params, mode="all")
    WarpLoader = DataLoader(dataset, 1, shuffle=False, num_workers=4)

    cache_save_folder = os.path.join(params.attribute_cache_path, 'cache_list')
    os.makedirs(cache_save_folder, exist_ok=True)

    vis_save_folder = os.path.join(params.attribute_cache_path, 'mask')
    # if params.splitName is not None:
    #     vis_save_folder = os.path.join(params.datasetFolder, params.modelName, params.splitName, "normal")
    # else:
    #     vis_save_folder = os.path.join(params.datasetFolder, params.modelName, "normal")
    os.makedirs(vis_save_folder, exist_ok=True)

    uvs = dataset.uv.to(params.device)
    verts = dataset.v.to(params.device)
    faces = dataset.f.to(params.device)
    normals = dataset.vn.to(params.device)
    centers = dataset.center.to(params.device)
    scales = dataset.scale.to(params.device)
    torch.save(dataset.center, os.path.join(cache_save_folder, "center.pt"))
    torch.save(dataset.scale, os.path.join(cache_save_folder, "scale.pt"))

    glctx = dr.RasterizeGLContext()
    with torch.no_grad():
        for step, (view_inds, sample) in tqdm(enumerate(WarpLoader, 0)):
            torch.cuda.empty_cache()
            mvp = sample['c2ws'].to(params.device)
            campos = sample['cpos'].to(params.device)
            images_ori_batch = sample['imgs'].to(params.device)
            if params.buffer_guidance:
                alb_ori_batch = sample['albedo'].to(params.device)
                ird_ori_batch = sample['irradiance'].to(params.device)
            num_batch_view = 1
                
            B, H, W, _ = images_ori_batch.shape
            v_pos_clip = torch.matmul(torch.nn.functional.pad(verts, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvp, 1, 2))
            rast, rast_db = dr.rasterize(glctx, v_pos_clip, faces, (H, W))  # shape: (N_v, H, W, 4)
            
            inlier_mask = (rast[..., 3] > 0).detach().cpu()
            
            # if params.undistort_crop_rate_h * params.undistort_crop_rate_w > 0:
            #     if params.undistort_crop_iter is not None and step >= params.undistort_crop_iter:
            #         temp_h, temp_w = images_ori_batch.shape[1], images_ori_batch.shape[2]
            #         crop_h = int(temp_h * params.undistort_crop_rate_h)
            #         crop_w = int(temp_w * params.undistort_crop_rate_w)
            #         inlier_mask = inlier_mask[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                
            vis_n = (inlier_mask[0].numpy() * 255).clip(0, 255)
            cv2.imwrite(os.path.join(vis_save_folder, "nrm_{:04}.png".format(params.all_view_list[view_inds])), vis_n.astype('uint8'))
            # breakpoint()