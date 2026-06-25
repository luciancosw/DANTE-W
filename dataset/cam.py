import os
import pdb
import imageio
from PIL import Image
from glob import glob
from readline import insert_text
import sys
from tkinter.messagebox import NO
import numpy as np
import torch
import cv2
import math
import trimesh
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
sys.path.append("../")
from configs.parameter import Params

flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def cv_to_gl(cv):
    gl = cv @ flip_mat  # convert to GL convention used in iNGP
    return gl

def mvsnet_to_dr(cameraKO4s, cameraPOs, sizes, ss_ratio, zn=0.1, zf=1000.0):
    mvps = []
    cs = []
    for v in range(cameraPOs.shape[0]):
        fl_x = cameraKO4s[v][0][0]
        c_x  = cameraKO4s[v][0][2]
        fl_y = cameraKO4s[v][1][1]
        c_y  = cameraKO4s[v][1][2]
        H, W = sizes[v] if len(sizes) > 1 else sizes[0]
        fov_x = math.atan(W * ss_ratio / (fl_x * 2)) * 2
        fov_y = math.atan(H * ss_ratio / (fl_y * 2)) * 2
        x = np.tan(fov_x / 2)
        y = np.tan(fov_y / 2)
        # aspect = W / H
        proj = np.array([[1/x,       0,        (W - 2*c_x)/W,            0], 
                         [           0, 1/-y,  (H - 2*c_y)/H,            0], 
                         [           0,    0, -(zf+zn)/(zf-zn), -(2*zf*zn)/(zf-zn)], 
                         [           0,    0,           -1,              0]], dtype=np.float32)
        out = cv2.decomposeProjectionMatrix(cameraPOs[v])
        R = out[1]
        t = out[2]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.transpose()
        c2w[:3, 3] = (t[:3] / t[3])[:, 0]
        
        c2w_gl = cv_to_gl(c2w)
        mv = np.linalg.inv(c2w_gl)
        campos = c2w_gl[:3, 3]
        mvp = proj @ mv
        mvps.append(mvp)
        cs.append(campos)
    mvps = np.stack(mvps, axis=0)
    cs = np.stack(cs, axis=0)
    return mvps, cs

def readCameraP0s_np_all(poseFolder, item_list):
    cameraPOs, cameraRTOs, cameraKOs = readCameraPOs_as_np(poseFolder, item_list)
    ones = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=cameraPOs.shape[0], axis=0)
    cameraP0s = np.concatenate((cameraPOs, ones), axis=1)
    return (cameraPOs, cameraP0s, cameraRTOs, cameraKOs)

def readCameraPOs_as_np(poseFolder, item_list):
    cameraPOs = np.empty((len(item_list), 3, 4), dtype=np.float64)
    cameraRTOs = np.empty((len(item_list), 3, 4), dtype=np.float64)
    cameraKOs = np.empty((len(item_list), 3, 3), dtype=np.float64)
    
    for _i, i_name in enumerate(item_list):
        _cameraPO, _cameraRT, _cameraK = readCameraP0_as_np_tanks(
            cameraPO_file=os.path.join(poseFolder, '{}_cam.txt'.format(i_name))
        )
        cameraPOs[_i] = _cameraPO
        cameraRTOs[_i] = _cameraRT
        cameraKOs[_i] = _cameraK
    return cameraPOs, cameraRTOs, cameraKOs

def readCameraP0_as_np_tanks(cameraPO_file):
    with open(cameraPO_file) as f:
        lines = f.readlines()
    cameraRTO = np.empty((3, 4)).astype(np.float64)
    cameraRTO[0, :] = np.array(lines[1].rstrip().split(' ')[:4], dtype=np.float64)
    cameraRTO[1, :] = np.array(lines[2].rstrip().split(' ')[:4], dtype=np.float64)
    cameraRTO[2, :] = np.array(lines[3].rstrip().split(' ')[:4], dtype=np.float64)

    cameraKO = np.empty((3, 3)).astype(np.float64)
    cameraKO[0, :] = np.array(lines[7].rstrip().split(' ')[:3], dtype=np.float64)
    cameraKO[1, :] = np.array(lines[8].rstrip().split(' ')[:3], dtype=np.float64)
    cameraKO[2, :] = np.array(lines[9].rstrip().split(' ')[:3], dtype=np.float64)

    cameraPO = np.dot(cameraKO, cameraRTO)
    return cameraPO, cameraRTO, cameraKO

class PatchDataset:
    def __init__(self, params, mode="train", only_mesh=False):
        super(PatchDataset, self).__init__()
        self.params = params
        if not only_mesh:
            if mode == "train":
                self.select_view_list = self.params.training_view_list
            elif mode == "all":
                self.select_view_list = self.params.all_view_list
            else:
                self.select_view_list = self.params.test_view_list
            
            imgPath = sorted(glob(self.params.imgNamePattern))
            item_list = [i.split('/')[-1].split('.')[0] for i in imgPath]
            item_list = [item_list[i] for i in self.select_view_list]
            
            # ------------------------------------------------------------------------------------------------------------------------
            # read cameras in mvsnet / colmap convention
            self.cameraPOs, self.cameraPO4s, \
            self.cameraRTO4s, self.cameraKO4s = readCameraP0s_np_all(
                poseFolder=self.params.poseFolder,
                item_list=item_list)
            if self.params.ss_ratio > 1:
                self.cameraKO4s[:, :2] *= self.params.ss_ratio
                self.cameraPOs = np.matmul(self.cameraKO4s, self.cameraRTO4s)
                self.cameraPO4s = np.concatenate((self.cameraPOs, np.repeat(np.array([[[0., 0., 0., 1.]]]), repeats=self.cameraPOs.shape[0], axis=0)), axis=1)
            # ------------------------------------------------------------------------------------------------------------------------
            # # convert cameras into open-gl convention used by nvdiffrast
            # self.cameraPO4s, self.cameraTs_new = mvsnet_to_dr(self.cameraKO4s, self.cameraPOs, self.sizes_all, self.params.ss_ratio, self.params.z_near, self.params.z_far)
            # ------------------------------------------------------------------------------------------------------------------------
    
    def __len__(self):
        return len(self.cameraPO4s)

    def __getitem__(self, index):
        """
        Returns:
            imgs: [B, H, W, 3]
            c2ws: [B, 4, 4]
            cpos: [B, 3]
        """
        ret = {}
        ret.update({
            "c2ws": torch.from_numpy(self.cameraPO4s[index]).float(),
            "cpos": torch.from_numpy(self.cameraTs_new[index]).float(),
        })
        return index, ret
    
if __name__ == "__main__":
    params = Params()
    pdata = PatchDataset(params=params)
    sample = pdata.__getitem__(12)
    for k, v in sample.items():
        print(k)
        print(v.shape, v.dtype)
    # breakpoint()