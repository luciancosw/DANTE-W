import sys
sys.path.append("../")
import random
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from configs.parameter import Params
from dataset.patch import PatchDataset
from models.model import NeuralHashRenderer
from models.ref_model import NeuralRefRenderer
from models.int_model import NeuralIntrinsicRenderer
from utils.utils import load_ckpt
from multiprocessing import Pool, cpu_count

def atlas_grid(resol=512):
    
    # Output resolution of the geometry image
    H, W = resol, resol

    # Prepare output image (float32, 3 channels for XYZ)
    geometry_img = np.zeros((H, W, 3), dtype=np.float32)
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    xy_grid = np.stack((x_grid, y_grid), axis=-1)
    uv_grid = np.stack([xy_grid[..., 0] / (W - 1), 1 - xy_grid[..., 1] / (H - 1)], axis=-1)
    
    return uv_grid

if __name__ == "__main__":
    tex_resol = 16384
    # tex_resol = 8192
    # tex_resol = 4096
    # tex_resol = 2048
    batch_size = 989 * 1320
    
    params = Params()
    if params.x_mode == "diffuse":
        model = NeuralHashRenderer(params=params).to(params.device)
    elif params.x_mode == "reflectance":
        model = NeuralRefRenderer(params=params).to(params.device)
    elif params.x_mode == "intrinsic":
        model = NeuralIntrinsicRenderer(params=params).to(params.device)
    
    load_ckpt(model, params.load_checkpoint_dir)
    
    if params.splitName is not None:
        render_save_folder = os.path.join(params.root_file, 'point_exp', params.modelName, params.splitName, 'texture_map')
    else:
        render_save_folder = os.path.join(params.root_file, 'point_exp', params.modelName, 'texture_map')
    os.makedirs(render_save_folder, exist_ok=True)
    
    uv_image = atlas_grid(tex_resol)
    uv_image = uv_image.reshape(-1, 2)
    albedo_image = []
    for batch_i in tqdm(range(0, uv_image.shape[0], batch_size)):
        geo_batch = torch.from_numpy(uv_image[batch_i: batch_i + batch_size]).float().to(params.device)
        with torch.no_grad():
            spatial_feat = model.spatial_enc(geo_batch)
            albedo_batch = model.texture_dec(spatial_feat).cpu().numpy()
        albedo_image.append(albedo_batch)
    albedo_image = np.concatenate(albedo_image, axis=0).reshape(tex_resol, tex_resol, -1)
    cv2.imwrite(os.path.join(render_save_folder, params.atlas_load_path.split('/')[-1].replace('.obj', '.jpg')), (albedo_image * 256).clip(0, 255).astype('uint8')[..., ::-1])

        
        
    