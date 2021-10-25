import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False
REQUIRES_LOGGING    = False

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(Criterion, self).__init__()
        self.n_classes          = opt.n_classes
        self.batchminer = batchminer
        self.name  = 'simclr'
        self.lr = opt.pred_lr
        self.temperature = opt.temperature

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.BatchNorm1d(opt.embed_dim),
                                       nn.Linear(opt.embed_dim, opt.pred_dim, bias=False),
                                       nn.BatchNorm1d(opt.pred_dim),
                                       nn.ReLU(inplace=True), # hidden layer
                                       nn.Linear(opt.pred_dim, opt.embed_dim)) # output layer

        self.criterion = torch.nn.CrossEntropyLoss()

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM
        self.REQUIRES_LOGGING = REQUIRES_LOGGING

    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """

        bs, dim = batch.size()
        labels = torch.cat([torch.ones(2) * i for i in range(int(bs/2))], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        similarity_matrix = torch.matmul(batch, batch.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(batch.device)

        logits = logits / self.temperature

        return self.criterion(logits, labels)