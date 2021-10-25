import torch
import copy
import wandb
import numpy as np

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True
REQUIRES_LOGGING    = True

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
        self.sampling_incl_pos = opt.loss_multisimilarity_sampling_incl_pos

        if 'pads' in self.sampling_mode:
            self.sampling_distr = Sampling_Distribution(opt.loss_multisimilarity_init_distr_nbins, [0., 1.],
                                                        update_step_size=25, ema_alpha=opt.loss_multisimilarity_sampling_ema_alpha,
                                                        update_type=self.sampling_mode, init_distr=opt.loss_multisimilarity_init_distr,
                                                        init_distr_mu=opt.loss_multisimilarity_init_distr_mu,
                                                        init_distr_sigma=opt.loss_multisimilarity_init_distr_sigma,
                                                        normalize_update=opt.loss_multisimilarity_sampling_normalize_update)
        else:
            self.sampling_distr = None

        #### margin loss params
        self.beta_const = opt.loss_multisimilarity_beta_const
        self.beta = torch.nn.Parameter(torch.ones(opt.n_classes)*opt.loss_multisimilarity_beta)
        self.lr = opt.loss_multisimilarity_beta_lr

        ####
        self.ALLOWED_MINING_OPS= ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM
        self.REQUIRES_LOGGING = REQUIRES_LOGGING

    def forward(self, batch, labels, global_step, **kwargs):

        assert batch.size(0) == labels.size(0), \
            f"batch.size(0): {batch.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = batch.size(0)
        sim_mat = batch.mm(batch.T)

        epsilon = 1e-5
        bs, dim = batch.shape
        loss = list()
        neg_losses = list()
        neg_bins = list()

        labels = labels.unsqueeze(1)
        bsame_labels = (labels.T == labels.view(-1,1)).to(batch.device).T
        bdiff_labels = (labels.T != labels.view(-1,1)).to(batch.device).T

        for i in range(batch_size):
            pos_ixs = copy.deepcopy(bsame_labels[i])
            pos_ixs[i] = False
            neg_ixs = copy.deepcopy(bdiff_labels[i])

            pos_pair_ = sim_mat[i][pos_ixs]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon] # exclude sample itself

            if self.sampling_incl_pos:
                neg_pair_ = sim_mat[i]
                neg_pair_ = neg_pair_[neg_pair_ < 1 - epsilon] # exclude sample itself 
            else:
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

            elif 'pads' in self.sampling_mode:
                pos_pair = pos_pair_[np.random.choice(len(pos_pair_))] # choose random positive
                q_d_inv, bins = self.sampling_distr.sample(neg_pair_)

                idx_tmp = np.random.choice(len(neg_pair_), p=q_d_inv)
                neg_pair = neg_pair_[idx_tmp]
                neg_bins.append(bins[idx_tmp])


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
                neg_losses.append(neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        if self.sampling_distr:
            if 'random' in self.sampling_distr.update_type:
                if global_step > 0 and global_step % self.sampling_distr.update_step_size == 0:
                    self.sampling_distr.update_sample_distr()
            elif 'ema' in self.sampling_distr.update_type:
                change_info = {'losses': neg_losses, 'bins': neg_bins}
                self.sampling_distr.update_sample_distr(change_info)

        loss = sum(loss) / batch_size
        return loss

    def inverse_sphere_cosine_distances(self, dim, anchor_to_all_dists):
            dists  = anchor_to_all_dists

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = (float(3 - dim) / 2) * torch.log(1.0 - dists.pow(2))
            q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()

    def get_log_data(self):
        if 'pads' in self.sampling_mode:
            return self.sampling_distr.log_sampling_distr()
        else:
            distr = self.inverse_sphere_cosine_distances(384, torch.range(0, 0.7, 0.77/25))
            support = np.linspace(0, 0.7, 24)
            distr_data = {'Sampling Distr': wandb.Histogram(np_histogram=(np.array(distr), np.array(support))),
                          'Log Sampling Distr': wandb.Histogram(np_histogram=(np.log(np.clip(np.array(distr), 1e-20, None)) - np.log(1e-20), np.array(support)))}

            return distr_data

class Sampling_Distribution():
    def __init__(self, n_support, support_limit, update_step_size, update_type, ema_alpha=0.25,
                 init_distr='random', init_distr_mu=0.6, init_distr_sigma=0.05, normalize_update=True,
                 filter_support_limit=False):

        self.n_support, self.support_limit = n_support, support_limit
        self.init_distr = init_distr
        self.init_distr_mu = init_distr_mu
        self.init_distr_sigma = init_distr_sigma
        self.normalize_update = normalize_update
        self.support = np.linspace(support_limit[0], support_limit[1], self.n_support)
        self.update_step_size = update_step_size
        self.update_type = update_type
        self.ema_alpha = ema_alpha

        self.no_norm_distr = self.init_sample_distr()
        self.distr = self.norm(self.no_norm_distr)

        self.distr_collect = {'support_limit': support_limit, 'n_support': n_support, 'progression': [], 'nonorm_progression': []}
        self.filter_support_limit = filter_support_limit


    def give_distr(self, name):
        if name   == 'random':
            distr_to_init = np.array([1.]*(self.n_support-1))
        elif name == 'distance':
            distr_to_init = self.probmode(self.support, self.n_support, upper_lim=0.5, mode='distance')
        elif name == 'uniform_low_and_tight':
            distr_to_init = self.probmode(self.support, self.n_support, mu=0.4, sig=0.1, mode='uniform')
        elif name == 'uniform_lowish_and_tight':
            distr_to_init = self.probmode(self.support, self.n_support, mu=0.6, sig=0.1, mode='uniform')
        elif name == 'heavyside_low':
            distr_to_init = self.probmode(self.support, self.n_support, mu=0.5,  mode='heavyside')
        elif name == 'uniform_low':
            distr_to_init = self.probmode(self.support, self.n_support, mu=0.5, sig=0.2, mode='uniform')
        elif name == 'uniform_avg':
            distr_to_init = self.probmode(self.support, self.n_support, mu=1.,  sig=0.2, mode='uniform')
        elif name == 'uniform_high':
            distr_to_init = self.probmode(self.support, self.n_support, mu=1.5, sig=0.2, mode='uniform')
        elif name == 'normal_low':
            # distr_to_init = self.probmode(self.support, self.n_support, mu=0.6, sig=0.05, mode='gaussian')
            distr_to_init = self.probmode(self.support, self.n_support, mu=self.init_distr_mu, sig=self.init_distr_sigma, mode='gaussian')
        elif name == 'normal_avg':
            distr_to_init = self.probmode(self.support, self.n_support, mu=1.1,  sig=0.04, mode='gaussian')
        elif name == 'normal_high':
            distr_to_init = self.probmode(self.support, self.n_support, mu=1.6, sig=0.04, mode='gaussian')
        else:
            raise Exception('Init. Distr. >> {} << not available!'.format(name))
        return distr_to_init

    def norm(self, distr):
        return distr/np.sum(distr)

    def init_sample_distr(self):
        if 'random' in self.update_type:
            no_norm_distr = np.random.choice(2, self.n_support - 1) + 0.05
        elif 'rl' in self.update_type:
            no_norm_distr = self.give_distr(self.init_distr)
        elif 'ema' in self.update_type:
            # no_norm_distr = np.array([1.]*(self.n_support-1))
            no_norm_distr = self.give_distr(self.init_distr)

        return no_norm_distr

    def update_sample_distr(self, change_info=None):

        if 'random' in self.update_type:
            self.no_norm_distr = np.random.choice(2, self.n_support) + 0.05
        elif 'rl' in self.update_type:
            self.no_norm_distr = np.clip(self.no_norm_distr * change_info['distr_change'], 1e-25, 1e25)
        elif 'ema' in self.update_type:
            distr_change = np.zeros_like(self.no_norm_distr)
            counter = np.zeros_like(self.no_norm_distr)
            for i,val in zip(change_info['bins'], change_info['losses']):
                distr_change[i] += val.cpu().detach().numpy()
                counter[i] += 1

            if self.normalize_update:
                counter[counter == 0] = 1
                distr_change = distr_change / counter
            self.no_norm_distr = (1-self.ema_alpha) * self.no_norm_distr + self.ema_alpha * distr_change
            self.no_norm_distr = np.clip(self.no_norm_distr, 1e-25, 1e25)

        # normalize
        self.distr = self.norm(self.no_norm_distr)

        # save history
        self.distr_collect['progression'].append(copy.deepcopy(self.distr))
        self.distr_collect['nonorm_progression'].append(copy.deepcopy(self.no_norm_distr))

    def sample(self, distances):
        p_assigns = np.sum((distances.cpu().detach().numpy().reshape(-1)>self.support[1:-1].reshape(-1,1)).T,axis=1).reshape(distances.shape)
        sample_p = self.distr[p_assigns]

        # if not self.filter_support_limit:
        #     outside_support_lim = (distances.cpu().numpy().reshape(-1)<self.support_limit[0]) * (distances.cpu().numpy().reshape(-1)>self.support_limit[1])
        #     outside_support_lim = outside_support_lim.reshape(distances.shape)
        #     sample_ps[outside_support_lim] = 0

        # if self.include_pos:
        #     sample_p = sample_ps[i]
        #     sample_p = sample_p/sample_p.sum()
        #     negatives.append(np.random.choice(bs,p=sample_p))
        # else:

        sample_p = sample_p/sample_p.sum()
        # neg = np.random.choice(np.arange(len(distances)), p=sample_p)

        return sample_p, p_assigns

    def log_sampling_distr(self):
        distr_data = dict()
        distr_data['Sampling Distr'] = wandb.Histogram(np_histogram=(np.array(self.distr),np.array(self.support)))
        distr_data['Log Sampling Distr'] = wandb.Histogram(np_histogram=(np.log(np.clip(np.array(self.distr), 1e-20, None))-np.log(1e-20),np.array(self.support)))

        return distr_data


    def gaussian(self, x, mu=1, sig=1):
        return 1/np.sqrt(2*np.pi*sig)*np.exp(-(x-mu)**2/(2*sig**2))

    def uniform(self, x, mu=1, sig=0.25):
        sp = x.copy()
        sp[(x>=mu-sig) * (x<=mu+sig)] = 1
        sp[(x<mu-sig) + (x>mu+sig)]   = 0
        return sp

    def distance(self, x, upper_lim):
        sp = x.copy()
        sp[x<=upper_lim] = 1
        sp[x>=upper_lim] = 0
        return sp

    def heavyside(self, x, mu):
        sp = x.copy()
        sp[x<=mu] = 1
        sp[x>mu]  = 0
        return sp

    def probmode(self, support_space, support, mu=None, sig=None, upper_lim=None, mode='uniform'):
        if mode=='uniform':
            probdist = self.uniform(support_space[:-1]+2/support, mu=mu, sig=sig)
        elif mode=='gaussian':
            probdist = self.gaussian(support_space[:-1]+2/support, mu=mu, sig=sig)
        elif mode=='distance':
            probdist = self.distance(support_space[:-1]+2/support, upper_lim)
        elif mode=='heavyside':
            probdist = self.heavyside(support_space[:-1]+2/support, mu=mu)
        probdist = np.clip(probdist, 1e-15, 1)
        probdist = probdist/probdist.sum()
        return probdist
