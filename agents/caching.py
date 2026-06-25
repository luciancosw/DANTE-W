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
    dataset = PatchDataset(params=params, mode="train")
    WarpLoader = DataLoader(dataset, 1, shuffle=False, num_workers=4)

    cache_save_folder = os.path.join(params.attribute_cache_path, 'cache_list')
    os.makedirs(cache_save_folder, exist_ok=True)

    vis_save_folder = os.path.join(params.attribute_cache_path, 'normal')
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
            num_batch_view = 1
                
            B, H, W, _ = images_ori_batch.shape
            v_pos_clip = torch.matmul(torch.nn.functional.pad(verts, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvp, 1, 2))
            rast, rast_db = dr.rasterize(glctx, v_pos_clip, faces, (H, W))  # shape: (N_v, H, W, 4)
            
            frg_xyz, _  = dr.interpolate(verts.unsqueeze(0), rast, faces)     # shape: (N_v, H, W, 3)
            frg_uv, _  = dr.interpolate(uvs.unsqueeze(0), rast, faces)     # shape: (N_v, H, W, 2)
            frg_normal, _ = dr.interpolate(normals.unsqueeze(0), rast, faces)  # shape: (N_v, H, W, 3)
            frg_normal = F.normalize(frg_normal, p=2, dim=-1, eps=1e-8).contiguous()
            frg_dir = frg_xyz - campos[:, None, None, :]                     # shape: (N_v, H, W, 3)
            frg_dir = F.normalize(frg_dir, p=2, dim=-1, eps=1e-8).contiguous()  # shape: (N_v, H, W, 3)
            inlier_mask = rast[..., 3:] > 0
            outlier_mask = ~inlier_mask.detach().cpu()
            
            if params.undistort_crop_rate_h * params.undistort_crop_rate_w > 0:
                if params.undistort_crop_iter is not None and step >= params.undistort_crop_iter:
                    temp_h, temp_w = images_ori_batch.shape[1], images_ori_batch.shape[2]
                    crop_h = int(temp_h * params.undistort_crop_rate_h)
                    crop_w = int(temp_w * params.undistort_crop_rate_w)
                    frg_uv = frg_uv[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                    frg_xyz = frg_xyz[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                    frg_normal = frg_normal[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                    frg_dir = frg_dir[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                    temp_h, temp_w = images_ori_batch.shape[1], images_ori_batch.shape[2]
                    crop_h = int(temp_h * params.undistort_crop_rate_h)
                    crop_w = int(temp_w * params.undistort_crop_rate_w)
                    images_ori_batch = images_ori_batch[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                    inlier_mask = inlier_mask[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                    if params.buffer_guidance:
                        alb_ori_batch = alb_ori_batch[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]

            colour = images_ori_batch[inlier_mask[..., 0]].detach().cpu()
            if params.buffer_guidance:
                albedo = alb_ori_batch[inlier_mask[..., 0]].detach().cpu()
                
            # vis_n = (images_ori_batch[0].cpu().numpy() * 256).clip(0, 255) * 0.7 + (((frg_normal + 1.0) * 0.5)[0].cpu().numpy() * 256).clip(0, 255) * 0.3
            vis_n = (alb_ori_batch[0].cpu().numpy() * 256).clip(0, 255) * 0.7 + (((frg_normal + 1.0) * 0.5)[0].cpu().numpy() * 256).clip(0, 255) * 0.3
            cv2.imwrite(os.path.join(vis_save_folder, "nrm_{}.jpg".format(params.training_view_list[view_inds])), vis_n.astype('uint8')[..., ::-1])
            # breakpoint()

            uv = frg_uv[inlier_mask[..., 0]].detach().cpu()
            xyz = frg_xyz[inlier_mask[..., 0]].detach().cpu()
            normal = frg_normal[inlier_mask[..., 0]].detach().cpu()
            viewdr = frg_dir[inlier_mask[..., 0]].detach().cpu()
            
            torch.save(uv, os.path.join(cache_save_folder, "uv_{}.pt".format(params.training_view_list[view_inds])))
            torch.save(xyz, os.path.join(cache_save_folder, "xyz_{}.pt".format(params.training_view_list[view_inds])))
            torch.save(normal, os.path.join(cache_save_folder, "normal_{}.pt".format(params.training_view_list[view_inds])))
            torch.save(colour, os.path.join(cache_save_folder, "colour_{}.pt".format(params.training_view_list[view_inds])))
            torch.save(viewdr, os.path.join(cache_save_folder, "viewdr_{}.pt".format(params.training_view_list[view_inds])))
            # breakpoint()
            if params.buffer_guidance:
                torch.save(albedo, os.path.join(cache_save_folder, "albedo_{}.pt".format(params.training_view_list[view_inds])))
                torch.save(irradn, os.path.join(cache_save_folder, "irradn_{}.pt".format(params.training_view_list[view_inds])))