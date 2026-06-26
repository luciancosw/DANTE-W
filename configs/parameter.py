import pdb

import numpy as np
import os
import torch
import math

class Params(object):
    def __init__(self):
        self.exp_id = '0625/exp_1'
        self.root_params()
        self.network_params()
        self.train_params()
        self.load_params()
        self.reconstruct_params()
        self.shading_params()
        self.cluster_params()
        self.render_params()

    def root_params(self):
        self.root_file = '/data/guangyu/aLit/record'
        self._input_data_rootFld = '/data/guangyu/dataset'

        self._debug_data_rootFld = os.path.join(self.root_file, 'debug', self.exp_id)
        self.summary_dir = os.path.join(self.root_file, 'experiment/train/log/')
        print('self.summary_dir', self.summary_dir)
        self.checkpoint_dir = os.path.join(self.root_file, 'experiment/train/state', self.exp_id)

        self.load_checkpoint_dir = None
        # self.load_checkpoint_dir = "/data/guangyu/aLit/record/experiment/train/state/0625/exp_1/epoch=19.ckpt"

    def network_params(self):
        self.ss_ratio = 1
        
        # -----------------------------------------------------------------------------------------------------------
        # network and embedding size.
        self.hash_base_resol = 16
        self.hash_n_levels = 16
        self.hash_per_level_scale = 1.7562521603732995  # 21 -> 2.0885475648548275, 20 -> 2.0, 19 -> 1.9152065613971474, 18 -> 1.8340080864093424, 17 -> 1.7562521603732995, 16 -> 1.681792830507429, 15 -> 1.6104903319492543, 14 -> 1.5422108254079407, 13 -> 1.4768261459394993, 12 -> 1.4142135623730951, 2048 -> 1.3542555469368927, 1024 -> 1.2968395546510096, 512 -> 1.241857812073484, 256 -> 1.1894969184913455, 128 -> 1.1393426577977728, 64 -> 1.0915649595217405
        self.hash_n_features_per_level = 2     # 8
        self.log2_hashmap_size = 22     # 22
        self.descriptor_dim = 64
        # -----------------------------------------------------------------------------------------------------------
        self.x_mode = "intrinsic" # 'diffuse', 'reflectance', 'intrinsic'
        self.buffer_guidance = True
        self.gray_shading = False
        self.app_model = False

    def train_params(self):
        self.use_cuda = True
        if (self.use_cuda and torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.max_epoch = 20    # ~10 epochs is enough, more is better
        # -----------------------------------------------------------------------------------------------------------
        # learning rate of network and embeddings.
        self.lr_net = 2e-3
        self.lr_emb = 2e-3
        # -----------------------------------------------------------------------------------------------------------
        # num of epoch to save loss and ckpt.
        self.log_epoch: int = 1      # 100
        self.save_checkpoint_epoch: int = 5    # 5
        # -----------------------------------------------------------------------------------------------------------
        # relative weight for different losses.
        self.loss_rgb_weight = 1e0
        self.loss_alb_weight = 1e0
        self.loss_ird_weight = 0.0
        # -----------------------------------------------------------------------------------------------------------

    def shading_params(self):
        pass

    def cluster_params(self):
        # -----------------------------------------------------------------------------------------------------------
        self.random_view_batch_size = 1
        self.random_ray_batch_size = 8192 * 8
        # -----------------------------------------------------------------------------------------------------------

    def render_params(self):
        self.z_near = 0.01
        self.z_far = 1000.0

    def train_strategy(self, epoch):
        pass

    def reconstruct_params(self):
        pass
    
    def load_params(self):
        self.modelName = "Pavilion_of_Prince_Teng"
        self.splitName = "data_noon"
        self.dsp_factor = 4
        input_mesh_resol = '1_tex'    # 1
        self.datasetFolder = os.path.join(self._input_data_rootFld, 'aLit')
        self.imgNamePattern = os.path.join(self.datasetFolder, self.modelName, self.splitName, "images_{}/*.JPG".format(self.dsp_factor))
        self.poseFolder = os.path.join(self.datasetFolder, self.modelName, self.splitName, "cams_{}".format(self.dsp_factor))
        self.litsFolder = os.path.join(self.datasetFolder, self.modelName, self.splitName)
        self.atlas_load_path = os.path.join(self.datasetFolder, self.modelName, "{}.obj".format(input_mesh_resol))    # tengwangge
        self.albedoNamePattern = os.path.join(self.datasetFolder, self.modelName, self.splitName, "albedo_{}/*.jpg".format(self.dsp_factor))
        self.attribute_cache_path = os.path.join(self.root_file, 'experiment/caches', self.modelName, self.splitName+'{}'.format(self.dsp_factor), 'ss_{}'.format(self.ss_ratio))
        
        # -----------------------------------------------------------------------------------------------------------
        # # data_noon
        self.num_lit = 1
        self.all_view_list = list(range(204))
        self.hold_out_list = []
        self.test_view_list = []
        # self.test_view_list = self.all_view_list
        # self.test_view_list = self.all_view_list[::7]
        self.training_view_list = [i for i in self.all_view_list if i not in self.test_view_list and i not in self.hold_out_list]

        self.undistort_crop_rate_h = 0    # 1 / 29
        self.undistort_crop_rate_w = 0
        self.undistort_crop_iter = None  # None
        
        self.trajectory_lit_id = 0

        # ---
        self.navigation_path = os.path.join(self.datasetFolder, self.modelName, 'lit.npy')
        self.navigation_H = 1080    # 1080, 3240, 4320
        self.navigation_W = 1920    # 1920, 5760, 7680
        self.navigation_focal = 2000.0  # 2000.0, 12000.0, 8000.0
        self.render_image_size = (self.navigation_H, self.navigation_W)  # the rendered output size
        self.image_size_single = torch.FloatTensor([[[self.navigation_W, self.navigation_H]]])  # the size of the input image