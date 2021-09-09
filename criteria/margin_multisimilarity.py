import numpy as np
import copy
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
import copy


"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt, **kwargs):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(Criterion, self).__init__()

        self.n_classes = opt.n_classes
        self.name = 'margin_multisimilarity'

        #### MS loss params
        self.pos_weight = opt.loss_multisimilarity_pos_weight
        self.neg_weight = opt.loss_multisimilarity_neg_weight
        self.margin = opt.loss_multisimilarity_margin
        self.pos_thresh = opt.loss_multisimilarity_pos_thresh
        self.neg_thresh = opt.loss_multisimilarity_neg_thresh
        self.d_mode = opt.loss_multisimilarity_d_mode
        self.base_mode = opt.loss_multisimilarity_base_mode
        self.sampling_mode = opt.loss_multisimilarity_sampling_mode

        #### margin loss params
        self.beta_const = opt.loss_multisimilarity_beta_const
        self.beta = torch.nn.Parameter(torch.ones(opt.n_classes)*opt.loss_multisimilarity_beta)
        self.lr = opt.loss_multisimilarity_beta_lr

        ####
        self.ALLOWED_MINING_OPS= ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):

        assert batch.size(0) == labels.size(0), \
            f"batch.size(0): {batch.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = batch.size(0)
        sim_mat = batch.mm(batch.T)

        epsilon = 1e-5
        bs, dim = batch.shape
        loss = list()

        labels = labels.unsqueeze(1)
        bsame_labels = (labels.T == labels.view(-1,1)).to(batch.device).T
        bdiff_labels = (labels.T != labels.view(-1,1)).to(batch.device).T

        for i in range(batch_size):
            pos_ixs = copy.deepcopy(bsame_labels[i])
            pos_ixs[i] = False
            neg_ixs = copy.deepcopy(bdiff_labels[i])

            pos_pair_ = sim_mat[i][pos_ixs]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][neg_ixs]

            if self.sampling_mode == 'max_min':
                neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    continue

            elif self.sampling_mode == 'distance':
                pos_pair = pos_pair_[np.random.choice(len(pos_pair_))] # choose random positive
                q_d_inv = self.inverse_sphere_cosine_distances(dim, neg_pair_)
                neg_pair = neg_pair_[np.random.choice(len(neg_pair_), p=q_d_inv)]

            # weighting step
            if self.beta_const:
                pos_loss = 1.0 / self.pos_weight * torch.log(
                    1 + torch.sum(torch.exp(-self.pos_weight * (pos_pair - self.pos_thresh)))) # make self.thresh learnable
                neg_loss = 1.0 / self.neg_weight * torch.log(
                    1 + torch.sum(torch.exp(self.neg_weight * (neg_pair - self.neg_thresh)))) # make self.thresh learnable
                loss.append(pos_loss + neg_loss)
            else:
                # learnable threshold
                pos_loss = 1.0 / self.pos_weight * torch.log(
                    1 + torch.sum(torch.exp(-self.pos_weight * (pos_pair - self.beta[labels[i]])))) # make self.thresh learnable
                neg_loss = 1.0 / self.neg_weight * torch.log(
                    1 + torch.sum(torch.exp(self.neg_weight * (neg_pair - self.beta[labels[i]])))) # make self.thresh learnable
                loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss

    def inverse_sphere_cosine_distances(self, dim, anchor_to_all_dists):
            dists  = anchor_to_all_dists

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = (float(3 - dim) / 2) * torch.log(1.0 - dists.pow(2))
            q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()

    def mult_gpu(pop_size, num_samples):
        """Use torch.Tensor.multinomial to generate indices on a GPU tensor."""
        p = torch.ones(pop_size, device="cuda") / pop_size
        return p.multinomial(num_samples=num_samples, replacement=False)
