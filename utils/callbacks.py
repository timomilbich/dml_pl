import os
from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from pytorch_lightning.utilities.distributed import rank_zero_only

import pprint
import wandb, imageio
import torch
import torchvision
import numpy as np
import pytorch_lightning as pl


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):

        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # print("Project config")
            # print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            # print("Lightning config")
            # print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                os.rename(self.logdir, dst)

    # @rank_zero_only
    # def on_train_end(self, trainer, pl_module):
    #     # trainer only saves last step if validation_step is not implemented
    #     # but we want to save in that case, too
    #     should_activate = trainer.is_overridden('validation_step')
    #     if should_activate:
    #         checkpoint_callbacks = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
    #         [c.on_validation_end(trainer, trainer.get_model()) for c in checkpoint_callbacks]

def EarlyStoppingPL(**args):
    # return EarlyStopping(monitor="val/accuracy", min_delta=0.0001, verbose=False)
    return EarlyStopping(**args, verbose=False)