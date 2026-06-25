import os
import torch
import shutil
import trimesh
import numpy as np

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)
    
def vis_pcd(verts, save_path):
    pcd = trimesh.PointCloud(verts)
    pcd.export(save_path)

def vis_mesh(verts, faces, save_path):
    mesh = trimesh.Trimesh(verts, faces)
    mesh.export(save_path)

def save_checkpoint(model, iter, ckpt_dir, is_best=True):
    # file_name = 'epoch:%s.pth.tar'%str(self.current_epoch).zfill(5)
    file_name = 'iter:%s.pth.tar'%str(iter)
    state = {
        'iteration': iter,
    }
    state.update({
        'model': model.state_dict(),
    })
    
    if not os.path.exists(os.path.join(ckpt_dir, '')):
        os.makedirs(os.path.join(ckpt_dir, ''))
    # Save the state
    torch.save(state, os.path.join(ckpt_dir, file_name))
    # If it is the best copy it to another file 'model_best.pth.tar'
    if is_best:
        shutil.copyfile(os.path.join(ckpt_dir, file_name),
                        os.path.join(ckpt_dir, 'best.pth.tar'))