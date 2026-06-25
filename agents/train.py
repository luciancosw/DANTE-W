import sys
sys.path.append("../")
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from utils.utils import save_checkpoint
from tensorboardX import SummaryWriter
from configs.parameter import Params
from dataset.patch import PatchDataset
from models.model import NeuralHashRenderer

def update_lr(optimizer, epoch, config):
    learning_factor = (np.cos(np.pi * epoch / config.max_epoch) + 1.0) * 0.5 * (1 - 0.001) + 0.001
    for param_group in optimizer.param_groups:
        if "net" in param_group['name']:
            param_group['lr'] = config.lr_net * learning_factor
        if "emb" in param_group['name']:
            param_group['lr'] = config.lr_emb * learning_factor

if __name__ == "__main__":
    params = Params()
    dataset = PatchDataset(params=params)
    verts = dataset.v.unsqueeze(0).to(params.device)
    faces = dataset.f.to(params.device)
    normals = dataset.vn.unsqueeze(0).to(params.device)
    centers = dataset.center.unsqueeze(0).to(params.device)
    scales = dataset.scale.unsqueeze(0).to(params.device)
    TrainLoader = DataLoader(dataset, params.random_view_batch_size, shuffle=True, num_workers=4)

    model = NeuralHashRenderer(params=params)
    model.train()
    model.to(params.device)
    optimizer = model.config_optimizer()

    summary_writer = SummaryWriter(log_dir=params.summary_dir, comment='biscale')

    global_step = 0
    for epoch in range(params.max_epoch):
        loss_list = []

        if epoch % params.save_checkpoint_epoch == 0:
            save_checkpoint(model=model, iter=epoch, ckpt_dir=params.checkpoint_dir)

        # if params.progressive_epoch > 0:
        #     if params.spatial_enc_type == "hash":
        #         n_levels = min(epoch // params.progressive_epoch + params.hash_n_levels // 2, params.hash_n_levels)
        #         model.length = n_levels * params.hash_n_features_per_level

        torch.cuda.empty_cache()
        for step, (view_inds, sample) in tqdm(enumerate(TrainLoader, 0)):
            for k, v in sample.items():
                sample[k] = sample[k].to(params.device)

            B = sample['imgs'].shape[0]
            predict_rgb, predict_nrm, predict_mask, _ = model(
                sample=sample, 
                verts=verts.expand(B, -1, -1).contiguous(), 
                faces=faces.contiguous(), 
                normals=normals.expand(B, -1, -1).contiguous(), 
                centers=centers.expand(B, -1).contiguous(), 
                scales=scales.expand(B, -1).contiguous(),
            )

            # loss
            loss_rgb = (predict_rgb[predict_mask[..., 0]] - sample['imgs'][predict_mask[..., 0]]).abs().mean()
            loss = params.loss_rgb_weight * loss_rgb
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        if epoch % params.log_epoch == 0:
            summary_writer.add_scalar("iter/loss_rgb", loss_rgb, epoch)
        
        print(f"L={np.mean(loss_list):.4f}")
        update_lr(optimizer, epoch, params)
        

            