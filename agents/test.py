"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import sys

sys.path.append("../")
from configs.parameter import Params
from dataset.patch import PatchDataset
from models.model import NeuralHashRenderer
from models.ref_model import NeuralRefRenderer
from models.int_model import NeuralIntrinsicRenderer

from tensorboardX import SummaryWriter
import pdb
import time
import shutil
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr

from PIL import Image, ImageDraw
from tqdm import tqdm
from glob import glob

import random
import os

import numpy as np
import cv2
import scipy.misc
import math
import matplotlib.pyplot as plt
import time
from dataset.patch import mvsnet_to_dr
from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils.utils import load_ckpt


img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """
    def __init__(self, params):
        self.params = params
        self.logger = logging.getLogger("Agent")
        self.debug_flag = True

        self.t_begin = time.time()
        self.manual_seed = random.randint(1, 10000)
        print("seed: ", self.manual_seed)
        random.seed(self.manual_seed)

        use_cuda = self.params.use_cuda

        if (use_cuda and torch.cuda.is_available()):
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.manual_seed)
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            print("Program will run on *****CPU***** ")

        torch.multiprocessing.set_sharing_strategy('file_system')

        self.current_epoch = 0
        self.current_iteration = 0
        self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir, comment='biscale')

        if self.params.x_mode == "diffuse":
            self.GlobalModel = NeuralHashRenderer(params=self.params).to(self.device)
        elif self.params.x_mode == "reflectance":
            self.GlobalModel = NeuralRefRenderer(params=self.params).to(self.device)
        elif self.params.x_mode == "intrinsic":
            self.GlobalModel = NeuralIntrinsicRenderer(params=self.params).to(self.device)

        load_ckpt(self.GlobalModel, self.params.load_checkpoint_dir)
    
    def navigation_render(self, return_img=True):
        self.AtlasDataset = PatchDataset(params=self.params, mode="test", only_mesh=True)
        f, cx, cy = self.params.navigation_focal, self.params.navigation_W / 2, self.params.navigation_H / 2
        H = self.params.navigation_H
        W = self.params.navigation_W
        
        # ----------------------------------------------------------------------------------------------------------
        fov = np.radians(39.6)
        f = fov2focal(fov, W)
        # ----------------------------------------------------------------------------------------------------------
        
        # raster_size = torch.FloatTensor([[[W, H]]])
        intrinsics = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        if self.params.ss_ratio > 1:
            intrinsics[:2, :] *= self.params.ss_ratio
        
        extrinsics = np.load(self.params.navigation_path)
        extrinsics[:, :, 1] *= -1
        extrinsics = extrinsics[:, :, [0, 2, 1, 3]]
        intrinsics = intrinsics[None, ...].repeat(extrinsics.shape[0], axis=0)
        w2cs = np.matmul(intrinsics, extrinsics)
        c2ws, cpos = mvsnet_to_dr(intrinsics, w2cs, [[H, W]], self.params.ss_ratio, self.params.z_near, self.params.z_far)
        c2ws = torch.from_numpy(c2ws).type(torch.FloatTensor)
        cpos = torch.from_numpy(cpos).type(torch.FloatTensor)
        # c2ws = c2ws[:800]
        # cpos = cpos[:800]

        uvs = self.AtlasDataset.uv.unsqueeze(0).to(self.device)
        verts = self.AtlasDataset.v.unsqueeze(0).to(self.device)
        faces = self.AtlasDataset.f.to(self.device)
        normals = self.AtlasDataset.vn.unsqueeze(0).to(self.device)
        centers = self.AtlasDataset.center.unsqueeze(0).to(self.device)
        scales = self.AtlasDataset.scale.unsqueeze(0).to(self.device)
        
        N_v = c2ws.shape[0]
        with torch.no_grad():
            gif_colour = []
            gif_normal = []
            if self.params.x_mode == "intrinsic":
                gif_albedo = []
                gif_shading = []
            for v in tqdm(range(N_v)):                
                sample = {
                    'imgs': torch.zeros(1, H, W, 3).to(self.device),
                    'c2ws': c2ws[v:v+1].to(self.device),
                    'cpos': cpos[v:v+1].to(self.device)
                }
                
                ret = self.GlobalModel(
                    sample=sample, 
                    uvs=uvs.contiguous(), 
                    verts=verts.contiguous(), 
                    faces=faces.contiguous(), 
                    normals=normals.contiguous(), 
                    centers=centers.contiguous(), 
                    scales=scales.contiguous(),
                )
                
                pred_colour = ret['predict_rgb']
                pred_normal = ret['frg_normal']
                outlier_mask = ret['outlier_mask']
                
                pred_colour[outlier_mask[..., 0]] = 1.0
                pred_normal[outlier_mask[..., 0]] = 1.0
                pred_colour = pred_colour[0].detach().cpu()
                pred_normal = pred_normal[0].detach().cpu()
                pred_colour = torch.clamp(pred_colour, min=0.0, max=1.0)
                pred_normal = (pred_normal + 1.0) * 0.5
                
                if self.params.x_mode == "intrinsic":
                    pred_albedo = ret['predict_albedo']
                    pred_shading = ret['predict_shading']
                    
                    pred_albedo[outlier_mask[..., 0]] = 1.0
                    pred_shading[outlier_mask[..., 0]] = 1.0
                    
                    pred_albedo = pred_albedo[0].detach().cpu()
                    pred_shading = pred_shading[0].detach().cpu()
                
                    pred_albedo = torch.clamp(pred_albedo, min=0.0, max=1.0)
                    pred_shading = torch.clamp(pred_shading, min=0.0, max=1.0)
                    
                    # pred_albedo = torch.pow(pred_albedo, 1.0 / 2.2)
                    # pred_shading = torch.pow(pred_shading, 1.0 / 2.2)
                    
                    gif_albedo.append((pred_albedo.numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
                    gif_shading.append((pred_shading.numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
            
                gif_colour.append((pred_colour.numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
                gif_normal.append((pred_normal.numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
        
        if self.params.splitName is not None:
            render_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, self.params.splitName, 'nv')
        else:
            render_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, 'nv')
        os.makedirs(render_save_folder, exist_ok=True)
        
        if return_img:
            render_img_folder = os.path.join(render_save_folder, "images")
            os.makedirs(render_img_folder, exist_ok=True)
            for i, image in enumerate(gif_colour):
                cv2.imwrite(os.path.join(render_img_folder, 'colour_v{}_l{}.png'.format(i, self.params.trajectory_lit_id)), image)
            if self.params.x_mode == "intrinsic":
                for i, image in enumerate(gif_albedo):
                    cv2.imwrite(os.path.join(render_img_folder, 'albedo_v{}_l{}.png'.format(i, self.params.trajectory_lit_id)), image)
                for i, image in enumerate(gif_shading):
                    cv2.imwrite(os.path.join(render_img_folder, 'shading_v{}_l{}.png'.format(i, self.params.trajectory_lit_id)), image)
                
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w, _ = gif_colour[0].shape
            writer = cv2.VideoWriter(os.path.join(render_save_folder, '{}_colour.mp4'.format(
                self.params.navigation_path.split('/')[-1].split('.')[0])), fourcc, 30, (w, h))
            for image in gif_colour:
                # print(image.shape)
                writer.write(image)
            writer.release()
            
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # h, w, _ = gif_colour[0].shape
            # writer = cv2.VideoWriter(os.path.join(render_save_folder, '{}_normal.mp4'.format(
            #     self.params.navigation_path.split('/')[-1].split('.')[0])), fourcc, 30, (w, h))
            # for image in gif_normal:
            #     # print(image.shape)
            #     writer.write(image)
            # writer.release()
            
            if self.params.x_mode == "intrinsic":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w, _ = gif_colour[0].shape
                writer = cv2.VideoWriter(os.path.join(render_save_folder, '{}_albedo.mp4'.format(
                    self.params.navigation_path.split('/')[-1].split('.')[0])), fourcc, 30, (w, h))
                for image in gif_albedo:
                    # print(image.shape)
                    writer.write(image)
                writer.release()
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w, _ = gif_colour[0].shape
                writer = cv2.VideoWriter(os.path.join(render_save_folder, '{}_irradiance.mp4'.format(
                    self.params.navigation_path.split('/')[-1].split('.')[0])), fourcc, 30, (w, h))
                for image in gif_shading:
                    # print(image.shape)
                    writer.write(image)
                writer.release()
            
    def test_render(self, ):        
        self.AtlasDataset = PatchDataset(params=self.params, mode="test")
        self.TestDataset = DataLoader(
            self.AtlasDataset,
            batch_size=5,
            shuffle=False, num_workers=4
        )
        
        if self.params.splitName is not None:
            render_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, self.params.splitName, 'test')
            if self.params.x_mode == "intrinsic":
                albedo_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, self.params.splitName, 'albedo')
                shading_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, self.params.splitName, 'irradiance')
                reflectance_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, self.params.splitName, 'reflectance')
        else:
            render_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, 'test')
            if self.params.x_mode == "intrinsic":
                albedo_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, 'albedo')
                shading_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, 'irradiance')
                reflectance_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, 'reflectance')
                
        os.makedirs(render_save_folder, exist_ok=True)
        if self.params.x_mode == "intrinsic":
            os.makedirs(albedo_save_folder, exist_ok=True)
            os.makedirs(shading_save_folder, exist_ok=True)
            os.makedirs(reflectance_save_folder, exist_ok=True)
        
        uvs = self.AtlasDataset.uv.unsqueeze(0).to(self.device)
        verts = self.AtlasDataset.v.unsqueeze(0).to(self.device)
        faces = self.AtlasDataset.f.to(self.device)
        normals = self.AtlasDataset.vn.unsqueeze(0).to(self.device)
        centers = self.AtlasDataset.center.unsqueeze(0).to(self.device)
        scales = self.AtlasDataset.scale.unsqueeze(0).to(self.device)

        with torch.no_grad():
            for index_data, (view_inds, sample) in tqdm(enumerate(self.TestDataset, 0)):
                B, H, W, _ = sample['imgs'].shape
                
                for k, v in sample.items():
                    sample[k] = sample[k].to(self.device)
                
                ret = self.GlobalModel(
                    sample=sample, 
                    uvs=uvs.expand(B, -1, -1).contiguous(), 
                    verts=verts.expand(B, -1, -1).contiguous(), 
                    faces=faces.contiguous(), 
                    normals=normals.expand(B, -1, -1).contiguous(), 
                    centers=centers.expand(B, -1).contiguous(), 
                    scales=scales.expand(B, -1).contiguous(),
                )
                
                pred_colour = ret['predict_rgb']
                pred_normal = ret['frg_normal']
                outlier_mask = ret['outlier_mask']

                pred_colour = torch.clamp(pred_colour, min=0.0, max=1.0)
                pred_normal = (pred_normal + 1.0) * 0.5

                pred_colour[outlier_mask[..., 0]] = 1.0
                pred_normal[outlier_mask[..., 0]] = 1.0
                
                gt_colour = sample['imgs']
                gt_colour[outlier_mask[..., 0]] = 1.0
                
                if self.params.x_mode == "intrinsic":
                    pred_albedo = ret['predict_albedo']
                    pred_shading = ret['predict_shading']
                    
                    pred_albedo = torch.clamp(pred_albedo, min=0.0, max=1.0)
                    pred_shading = torch.clamp(pred_shading, min=0.0, max=1.0)
                    
                    # pred_albedo = torch.pow(pred_albedo, 1.0 / 2.2)
                    # pred_shading = torch.pow(pred_shading, 1.0 / 2.2)
                    
                    pred_albedo[outlier_mask[..., 0]] = 1.0
                    pred_shading[outlier_mask[..., 0]] = 1.0
                    
                    if 'predict_reflectance' in ret.keys():
                        pred_reflect = ret['predict_reflectance']
                        pred_reflect = torch.clamp(pred_reflect, min=0.0, max=1.0)
                        pred_reflect = torch.pow(pred_reflect, 1.0 / 2.2)
                        pred_reflect[outlier_mask[..., 0]] = 1.0
                    
                for v in range(B):
                    # cv2.imwrite(os.path.join(render_save_folder, 'gt_{}.jpg'.format(self.params.test_view_list[view_inds[v]])), (gt_colour[v].cpu().numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
                    cv2.imwrite(os.path.join(render_save_folder, 'colour_{}.png'.format(self.params.test_view_list[view_inds[v]])), (pred_colour[v].cpu().numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
                    # cv2.imwrite(os.path.join(render_save_folder, 'normal_{}.jpg'.format(self.params.test_view_list[view_inds[v]])), (pred_normal[v].cpu().numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
                    if self.params.x_mode == "intrinsic":
                        cv2.imwrite(os.path.join(albedo_save_folder, '{}.png'.format(self.params.test_view_list[view_inds[v]])), (pred_albedo[v].cpu().numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
                        cv2.imwrite(os.path.join(shading_save_folder, '{}.png'.format(self.params.test_view_list[view_inds[v]])), (pred_shading[v].cpu().numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
                        if 'predict_reflectance' in ret.keys():
                            cv2.imwrite(os.path.join(reflectance_save_folder, '{}.png'.format(self.params.test_view_list[view_inds[v]])), (pred_reflect[v].cpu().numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])

if __name__ == '__main__':
    params = Params()
    agent = BaseAgent(params)
    print('start render #@~#@#@#!@#@!#')

    agent.test_render()
    # agent.navigation_render()

