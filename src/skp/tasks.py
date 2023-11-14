import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from . import builder
from . import optim
from . import metrics as pl_metrics
from .data.mixaug import apply_mixaug


class BaseTask(pl.LightningModule): 

    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model 

        self.val_loss = []
        
        self.save_hyperparameters(cfg)
        
    def set(self, name, attr):
        if name == 'metrics':
            attr = nn.ModuleList(attr) 
        setattr(self, name, attr)
    
    def on_train_start(self): 
        for obj in ['optimizer','scheduler','loss_fn','metrics','valid_metric']:
            assert hasattr(self, obj)

        self.total_training_steps = self.trainer.num_training_batches * self.trainer.max_epochs 
        self.current_training_step = 0
        
        if isinstance(self.scheduler, optim.CosineAnnealingLR):
            self.scheduler.T_max = self.total_training_steps 

        if isinstance(self.scheduler, optim.CustomOneCycleLR):
            self.scheduler.total_steps = self.total_training_steps
            self.scheduler.step_size_up = float(self.scheduler.pct_start * self.scheduler.total_training_steps) - 1
            self.scheduler.step_size_down = float(self.scheduler.total_training_steps - self.scheduler.step_size_up) - 1

    def training_step(self, batch, batch_idx):             
        X, y = batch
        if hasattr(self, 'mixaug') and isinstance(self.mixaug, dict):
            X, y = apply_mixaug(X, y, self.mixaug)
        p = self.model(X) 
        loss = self.loss_fn(p, y)
        self.log('loss', loss) 
        self.current_training_step += 1
        return loss

    def validation_step(self, batch, batch_idx): 
        X, y = batch
        p = self.model(X) 
        loss = self.loss_fn(p, y)
        self.val_loss += [loss]
        for m in self.metrics: m.update(p, y)
        return loss
        
    def validation_epoch_end(self, *args, **kwargs):
        metrics = {}
        for m in self.metrics:
            metrics.update(m.compute())
        metrics['val_loss'] = torch.stack(self.val_loss).mean() ; self.val_loss = []
        max_strlen = max([len(k) for k in metrics.keys()])

        if isinstance(self.valid_metric, list):
            metrics['vm'] = torch.sum(torch.stack([metrics[_vm.lower()].cpu() for _vm in self.valid_metric]))
        else:
            metrics['vm'] = metrics[self.valid_metric.lower()]

        self.log_dict(metrics)
        for m in self.metrics: m.reset()

        if self.global_rank == 0:
            print('\n========')
            for k,v in metrics.items(): 
                print(f'{k.ljust(max_strlen)} | {v.item():.4f}')

    def configure_optimizers(self):
        lr_scheduler = {
            'scheduler': self.scheduler,
            'interval': 'step' if self.scheduler.update_frequency == 'on_batch' else 'epoch'
        }
        if isinstance(self.scheduler, optim.ReduceLROnPlateau): 
            lr_scheduler['monitor'] = self.valid_metric
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': lr_scheduler
            }

    def train_dataloader(self):
        return builder.build_dataloader(self.cfg, self.train_dataset, 'train')

    def val_dataloader(self):
        return builder.build_dataloader(self.cfg, self.valid_dataset, 'valid')


class ClassificationTask(BaseTask):

    def __init__(self, cfg, model, mixaug=None):
        super().__init__(cfg, model)
        self.mixaug = mixaug


class SampleCadenceTask(ClassificationTask):

    def validation_step(self, batch, batch_idx): 
        X, y = batch
        # X.shape = (N, C, H, W)
        cadences = []
        channels = [0,1,2] if X.size(1) == 3 else [0,2,4]
        for c in channels:
            cadences += [self.model(X[:,c].unsqueeze(1)).sigmoid()]
        p = torch.stack(cadences, dim=0).mean(0)
        loss = self.loss_fn(p, y)
        self.val_loss += [loss]
        for m in self.metrics: m.update(p, y)
        return loss