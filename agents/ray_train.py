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
# from tensorboardX import SummaryWriter
from configs.parameter import Params
from dataset.ray import RayDataset
from models.model import NeuralHashRenderer
from models.ref_model import NeuralRefRenderer
from models.int_model import NeuralIntrinsicRenderer
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.utils import load_ckpt

class LightMetaSurfelSystem(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        if self.params.x_mode == "diffuse":
            self.model = NeuralHashRenderer(params=params)
        elif self.params.x_mode == "reflectance":
            self.model = NeuralRefRenderer(params=params)
        elif self.params.x_mode == "intrinsic":
            self.model = NeuralIntrinsicRenderer(params=params)
        if self.params.load_checkpoint_dir is not None:
            print('Loading ckpt from {}'.format(self.params.load_checkpoint_dir))
            load_ckpt(self.model, self.params.load_checkpoint_dir)
        self.model.train()
        self.train_dataset = RayDataset(params=params)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
    
    def forward(self, batch, ):
        return self.model.render_rays(batch)

    def configure_optimizers(self):
        opts = []
        self.net_opt = self.model.config_optimizer()
        opts += [self.net_opt]
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.params.max_epoch,
                                    self.params.lr_emb/30)

        return opts, [net_sch]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb, *args):
        ret = self(batch)
        loss_rgb = (ret['predict_rgb'] - batch['colour']).abs().mean()
        loss = self.params.loss_rgb_weight * loss_rgb
        
        if self.params.buffer_guidance and self.params.x_mode == "intrinsic":
            loss_alb = (ret['predict_albedo'] - batch['albedo']).abs().mean()
            loss += self.params.loss_alb_weight * loss_alb
            

        with torch.no_grad():
            self.train_psnr(ret['predict_rgb'], batch['colour'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', self.train_psnr, True)
        return loss

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

if __name__ == '__main__':
    params = Params()
    system = LightMetaSurfelSystem(params)

    ckpt_cb = ModelCheckpoint(dirpath=params.checkpoint_dir,
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=params.save_checkpoint_epoch,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=params.summary_dir,
                               name=params.exp_id,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=params.max_epoch,
                      check_val_every_n_epoch=params.max_epoch,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=1,
                      strategy="ddp",
                      num_sanity_val_steps=0,
                      precision=16)

    trainer.fit(system, ckpt_path=None)

