import sys

sys.path.append("../")
# from tensorboardX import SummaryWriter
import pdb
import time
import shutil
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw

from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os

import numpy as np
import cv2
import scipy.misc
import math
import matplotlib.pyplot as plt
import nvdiffrast.torch as dr
import time
try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()
from models.ref_model import reflectance_direction

class NeuralIntrinsicRenderer(nn.Module):
    def __init__(self, params):
        super(NeuralIntrinsicRenderer, self).__init__()
        self.params = params
        self.device = self.params.device

        self.spatial_enc = \
            tcnn.Encoding(
                2, 
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": self.params.hash_n_levels,
                    "n_features_per_level": self.params.hash_n_features_per_level,
                    "log2_hashmap_size": self.params.log2_hashmap_size,
                    "base_resolution": self.params.hash_base_resol,
                    "per_level_scale": self.params.hash_per_level_scale,
                    "interpolation": "Linear",  # "Smoothstep"
                }
            )
        
        self.lighting_enc = \
            tcnn.Encoding(
                3, 
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": self.params.hash_n_levels,
                    "n_features_per_level": self.params.hash_n_features_per_level,
                    "log2_hashmap_size": self.params.log2_hashmap_size,
                    "base_resolution": self.params.hash_base_resol,
                    "per_level_scale": 1.241857812073484,  # 1.046284092224702, 1.1393426577977728, 1.241857812073484
                    "interpolation": "Linear",  # "Smoothstep"
                }
            )
        
        shading_channels = 1 if self.params.gray_shading else 3
        self.shading_dec = \
            tcnn.Network(
                self.params.hash_n_levels * self.params.hash_n_features_per_level, 
                shading_channels, 
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": self.params.descriptor_dim,
                    "n_hidden_layers": 1,
                }
            )
    
        self.texture_dec = \
            tcnn.Network(
                self.params.hash_n_levels * self.params.hash_n_features_per_level, 
                3, 
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": self.params.descriptor_dim,
                    "n_hidden_layers": 1,
                }
            )
        
        self.albedo_ab = nn.Parameter(torch.zeros(1, 2).to(self.device))
        if self.params.app_model:
            self.appear_ab = nn.Parameter(torch.zeros(len(self.params.training_view_list), 2).to(self.device))
            
            
    def config_optimizer(self, ):
        params_to_train = [
            {'name': 'albedo_emb', 'params': self.spatial_enc.parameters(), 'lr': self.params.lr_emb},
            {'name': 'albedo_net', 'params': self.texture_dec.parameters(), 'lr': self.params.lr_net},
            {'name': 'lighting_emb', 'params': self.lighting_enc.parameters(), 'lr': self.params.lr_emb},
            {'name': 'shading_net', 'params': self.shading_dec.parameters(), 'lr': self.params.lr_net},
            # {'name': 'reflectance_net', 'params': self.reflectance_dec.parameters(), 'lr': self.params.lr_net},
            {'name': 'albedo_ab', 'params': self.albedo_ab, 'lr': self.params.lr_emb}
        ]
        if self.params.app_model:
            params_to_train += [{'name': 'appear_ab', 'params': self.appear_ab, 'lr': self.params.lr_emb}]
        return torch.optim.Adam(params_to_train)

    def render_rays(self, batch):
        """
        :param xyz: [N, 3]
        """
        uv, xyz, center, scale = batch['uv'], batch['xyz'], batch['center'], batch['scale']
        xyz -= center
        xyz /= scale
        xyz *= 2.0
        assert len(xyz.shape) == 2
        xyz = xyz * 0.5 + 0.5
        
        predict_shading = self.shading_dec(self.lighting_enc(xyz))
        predict_albedo = self.texture_dec(self.spatial_enc(uv))

        predict_rgb = linear_to_srgb((srgb_to_linear(predict_albedo).clip(0.0, 1.0) * torch.exp(self.albedo_ab[:, 0:1]) + self.albedo_ab[:, 1:]) * predict_shading).clip(0.0, 1.0)
        # predict_rgb = (predict_albedo * torch.exp(self.albedo_ab[:, 0:1]) + self.albedo_ab[:, 1:]) * predict_shading

        # predict_rgb = predict_albedo * predict_shading
        # predict_rgb = linear_to_srgb(predict_rgb).clip(0.0, 1.0)

        if self.params.app_model:
            appear_ab = self.appear_ab[batch['vid']]
            predict_rgb = torch.exp(appear_ab[:, 0:1]) * predict_rgb + appear_ab[:, 1:]
        
        return {
            'predict_rgb': predict_rgb,
            'predict_albedo': predict_albedo,
            'predict_shading': predict_shading, 
        }

    def forward(self, sample, uvs, verts, faces, normals, centers, scales,):
        """
        :param imgs: [B, H, W, 3]
        :param c2ws: [B, 4, 4]
        :param cpos: [B, 3]
        :param verts: [B, V, 3]
        :param faces: [B, F, 3]
        :param normals: [B, V, 3]
        :param centers: [B, 3]
        :param scales: [B, 3]
        """
        glctx = dr.RasterizeGLContext()
        B, H, W, _ = sample['imgs'].shape
        # assert B == 1
        v_pos_clip = torch.matmul(torch.nn.functional.pad(verts, pad=(0,1), mode='constant', value=1.0), torch.transpose(sample['c2ws'], 1, 2))
        rast, rast_db = dr.rasterize(glctx, v_pos_clip.float(), faces, (H * self.params.ss_ratio, W * self.params.ss_ratio))  # [N_v, H, W, 4]
        frg_xyz, _  = dr.interpolate(verts.float(), rast, faces)            # [N_v, H, W, 3]
        frg_uv, _  = dr.interpolate(uvs.float(), rast, faces)            # [N_v, H, W, 2]
        frg_normal, _ = dr.interpolate(normals.float(), rast, faces)        # [N_v, H, W, 3]
        frg_normal = F.normalize(frg_normal, p=2, dim=-1, eps=1e-8).contiguous()
        frg_dir = frg_xyz - sample['cpos'][:, None, None, :]                     # [N_v, H, W, 3]
        frg_dir = F.normalize(frg_dir, p=2, dim=-1, eps=1e-8).contiguous()       # [N_v, H, W, 3]
        inlier_mask = rast[..., 3:] > 0                                          # [N_v, H, W, 1]
        # inlier_mask = F.interpolate(inlier_mask.permute(0,3,1,2).float(), scale_factor=1.0 / self.params.ss_ratio, mode='bilinear', align_corners=True).permute(0,2,3,1) > 0.0
        outlier_mask = ~inlier_mask

        # import os, cv2
        # debug_save_path = "/data/guangyu/aLit/record/test_vis"
        # os.makedirs(debug_save_path, exist_ok=True)
        # for i in range(B):
        #     vis_m = ((sample['imgs'][i] * inlier_mask[i]).cpu().numpy() * 256).clip(0, 255)
        #     vis_n = (sample['imgs'][i].cpu().numpy() * 256).clip(0, 255) * 0.7 + (((frg_normal + 1.0) * 0.5)[i].cpu().numpy() * 256).clip(0, 255) * 0.3
        #     cv2.imwrite(os.path.join(debug_save_path, '{}_nrm.jpg'.format(i)), vis_n.astype('uint8')[..., ::-1])
        #     cv2.imwrite(os.path.join(debug_save_path, '{}_msk.jpg'.format(i)), vis_m.astype('uint8')[..., ::-1])
        # breakpoint()
                    
        center_pad = centers[:, None, None, :].contiguous()
        scale_pad = scales[:, None, None, :].contiguous()
        frg_xyz -= center_pad
        frg_xyz /= scale_pad
        frg_xyz *= 2.0
        
        B, H, W, C = frg_xyz.shape
        frg_xyz = frg_xyz.reshape(-1, C)
        frg_uv = frg_uv.reshape(-1, 2)
        
        frg_xyz_01 = frg_xyz * 0.5 + 0.5
        spatial_feat = self.spatial_enc(frg_uv)
        lighting_feat = self.lighting_enc(frg_xyz_01)
        neural_albedo = self.texture_dec(spatial_feat)
        neural_shading = self.shading_dec(lighting_feat)
        
        predict_albedo = neural_albedo.reshape(B, H, W, -1).float()
        predict_shading = neural_shading.reshape(B, H, W, -1).float()
        
        post_albedo = srgb_to_linear(predict_albedo).clip(0.0, 1.0) * torch.exp(self.albedo_ab[:, None, None, 0:1]) + self.albedo_ab[:, None, None, 1:]
        predict_rgb = linear_to_srgb(post_albedo * predict_shading).clip(0.0, 1.0)

        # post_albedo = predict_albedo * torch.exp(self.albedo_ab[:, None, None, 0:1]) + self.albedo_ab[:, None, None, 1:]
        # predict_rgb = post_albedo * predict_shading

        # predict_rgb = linear_to_srgb(predict_rgb).clip(0.0, 1.0)
        # predict_rgb = predict_rgb * (1 + 2 * 1e-3) - 1e-3
        return {
            'predict_rgb': predict_rgb,
            'predict_albedo': predict_albedo,
            'post_albedo': post_albedo,
            'predict_shading': predict_shading, 
            # 'predict_reflectance': predict_reflectance,
            'frg_normal': frg_normal,
            'inlier_mask': inlier_mask,
            'outlier_mask': outlier_mask
        }

def linear_to_srgb(linear):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.maximum(torch.tensor([1e-7]).to(linear.device), linear)**(5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)

def srgb_to_linear(srgb):
    linear0 = 25 / 323 * srgb

    base = torch.maximum(
        torch.tensor(1e-7, device=srgb.device, dtype=srgb.dtype),
        (200 * srgb + 11) / 211,
    )
    linear1 = base ** (12 / 5)

    threshold = 323 / 25 * 0.0031308
    return torch.where(srgb <= threshold, linear0, linear1)