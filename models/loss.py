import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        ## Initialize weightning factors for Loss

    def forward(self, outputs, labels, split="train"):

        ## Calculcate loss
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(outputs, labels)

        ## Log everything relevant in dict
        log = {"{}/loss".format(split): loss.clone().detach().mean()
              #,"{}/otherloss".format(split): accuracy.detach().mean()
              }
        return loss, log


