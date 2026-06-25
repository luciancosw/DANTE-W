import sys
sys.path.append("../")
import os
import numpy as np
import torch
import cv2
import math
import trimesh
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from configs.parameter import Params

class RayDataset:
    def __init__(self, params, mode="train"):
        super(RayDataset, self).__init__()
        self.params = params
        
        self.cache_file = os.path.join(self.params.attribute_cache_path, 'cache_list')
        if not os.path.exists(self.cache_file):
            raise ValueError('pre-cached data does not exist.')
        self.center = torch.load(os.path.join(self.cache_file, 'center.pt'))
        self.scale = torch.load(os.path.join(self.cache_file, 'scale.pt'))
        self.parse_caches()
        if self.params.buffer_guidance:
            self.parse_buffer()

    def parse_caches(self, ):
        self.vid, self.uv, self.xyz, self.normal, self.colour, self.viewdr = [], [], [], [], [], []

        def read_view_cache(v: int):
            uv = torch.load(os.path.join(self.cache_file, 'uv_{}.pt'.format(v)))
            xyz = torch.load(os.path.join(self.cache_file, 'xyz_{}.pt'.format(v)))
            normal = torch.load(os.path.join(self.cache_file, 'normal_{}.pt'.format(v)))
            colour = torch.load(os.path.join(self.cache_file, 'colour_{}.pt'.format(v)))
            viewdr = torch.load(os.path.join(self.cache_file, 'viewdr_{}.pt'.format(v)))
            return torch.tensor([self.params.training_view_list.index(v)]).expand(xyz.shape[0]), uv, xyz, normal, colour, viewdr

        self.vid, self.uv, self.xyz, self.normal, self.colour, self.viewdr = zip(*thread_map(read_view_cache, self.params.training_view_list, desc='Loading ViewCaches'))
        self.vid = torch.cat(self.vid, dim=0)
        self.uv = torch.cat(self.uv, dim=0)
        self.xyz = torch.cat(self.xyz, dim=0)
        self.normal = torch.cat(self.normal, dim=0)
        self.colour = torch.cat(self.colour, dim=0)
        self.viewdr = torch.cat(self.viewdr, dim=0)
    
    def parse_buffer(self, ):
        self.albedo = []
        
        def read_view_buffer(v: int):
            albedo = torch.load(os.path.join(self.cache_file, 'albedo_{}.pt'.format(v)))
            return albedo

        self.albedo = thread_map(read_view_buffer, self.params.training_view_list, desc='Loading ViewBuffer')
        self.albedo = torch.cat(self.albedo, dim=0)
        
    def __len__(self):
        return 1000

    def __getitem__(self, index):
        # randomly select pixels
        pix_idxs = np.random.choice(self.xyz.shape[0], self.params.random_ray_batch_size)
        sample = {}
        sample['vid'] = self.vid[pix_idxs]
        sample['uv'] = self.uv[pix_idxs]
        sample['xyz'] = self.xyz[pix_idxs]
        sample['normal'] = self.normal[pix_idxs]
        sample['colour'] = self.colour[pix_idxs]
        sample['viewdr'] = self.viewdr[pix_idxs]
        sample['center'] = self.center
        sample['scale'] = self.scale
        if self.params.buffer_guidance:
            sample['albedo'] = self.albedo[pix_idxs]
        return sample

if __name__ == "__main__":
    params = Params()
    dataset = RayDataset(params=params)
    print(dataset.xyz.shape)
    _, sample = dataset.__getitem__(666)
    for k, v in sample.items():
        print(k, v.shape)
