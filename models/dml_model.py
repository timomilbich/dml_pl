import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F
from utils.auxiliaries import instantiate_from_config
import numpy as np
import wandb
from criteria import add_criterion_optim_params

class DML_Model(pl.LightningModule):
    def __init__(self, config, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        config = OmegaConf.to_container(config)

        ## Init optimizer hyperparamters
        self.weight_decay = 0
        self.gamma = 0
        self.tau = 0

        ## Load model using config
        self.model = instantiate_from_config(config["Architecture"])

        ## Init loss
        batchminer = instantiate_from_config(config["Batchmining"]) if "Batchmining" in config.keys() else None
        config["Loss"]["params"]['batchminer'] = batchminer
        self.loss    = instantiate_from_config(config["Loss"])

        ## Init constom log scripts
        self.custom_logs = instantiate_from_config(config["CustomLogs"])

        if ckpt_path is not None:
            print("Loading model from {}".format(ckpt_path))
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        ## Load from checkpoint
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)

    def forward(self, x):
        out = self.model(x)
        x = out['embeds'] # {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y}
        return x

    def training_step(self, batch, batch_idx):
        ## Define one training step, the loss returned will be optimized
        inputs = batch[0]
        labels = batch[1]
        output = self.forward(inputs)

        loss, log_dict = self.loss(output, labels, split="train") ## Change inputs to loss
        self.log("Loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True) ## Add to progressbar
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False) ## Log in logger

        self.custom_logs.update(batch, output, mode="train") ##Class to wrap all other logging options
        self.log_dict(self.custom_logs.accuracy())

        return loss

    def validation_step(self, batch, batch_idx):
        ## Define one validation step, similar to training_step
        inputs = batch[0]
        labels = batch[1]

        with torch.no_grad():
            output = self.forward(inputs)
            loss, log_dict = self.loss(output, labels, split="val") ## Change inputs to loss

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True) ## Log in logger
        self.custom_logs.update(batch, output, mode="val") ##Class to wrap all other logging options
        self.log_dict(self.custom_logs.accuracy())
        self.logger.experiment.log(self.custom_logs.image_prediction(), commit = False)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, class_labels = batch
    #     with torch.no_grad():
    #         emb = self.forward(x)
    #
    #     return {"logits": emb, "labels": class_labels}

    # def validation_epoch_end(self, outputs):
    #     logits = torch.cat([x["logits"] for x in outputs]).cpu().detach()
    #     labels = torch.cat([x["labels"] for x in outputs]).cpu().detach()
    #
    #     sm = nn.LogSoftmax()
    #     correct = torch.argmax(sm(logits), dim=1) == labels
    #     accuracy = 100. * correct.sum() / labels.size()[0]
    #
    #     log_data = {
    #         "epoch": self.current_epoch,
    #         "val/accuracy": accuracy,
    #     }
    #     print(f"\nEpoch {self.current_epoch} validation: {accuracy:.2f}%")
    #     self.log_dict(log_data)


    def configure_optimizers(self):

        to_optim = [{'params': self.model.parameters(), 'lr': self.learning_rate, 'weight_decay': self.weight_decay}]
        to_optim = add_criterion_optim_params(self.loss, to_optim)
        optimizer = torch.optim.Adam(to_optim)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.tau, gamma=self.gamma)

        return [optimizer], [scheduler]


    def get_progress_bar_dict(self):
        ## Drop version name in progressbar
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict