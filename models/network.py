import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F
from utils.auxiliaries import instantiate_from_config
import numpy as np
import wandb

class Net(pl.LightningModule):
    def __init__(self, config, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        config = OmegaConf.to_container(config)

        ## Load models using config 
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        ## Init loss & custom log scripts with parameters in config
        self.loss    = instantiate_from_config(config["Loss"])
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
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
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
        ## Set up optimizers -- possible to add scheduler in second return argument
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.parameters()), lr=lr, betas=(0.9, 0.99))
        return [opt], []

    def get_progress_bar_dict(self):
        ## Drop version name in progressbar
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict