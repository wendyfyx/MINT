from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from chamferdist import ChamferDistance

class BaseLoss(ABC):
    
    @abstractmethod
    def __call__(self):
        pass
        

class BundleReconLoss(BaseLoss):
    def __init__(self, #feature_idx=[[0,1,2], [3]], 
                 recon_mode=['likelihood', 'likelihood'], 
                 recon_weight=[0.5, 0.5], **kwargs):
        # self.feature_idx = feature_idx
        self.recon_mode = recon_mode
        self.recon_weight = recon_weight

    def __call__(self, x, x_hat, **kwargs):
        # assert x.shape[-1] == len(sum(self.feature_idx, [])) \
        #     and x_hat.shape[-1] == len(sum(self.feature_idx, [])), \
        #     f"Shape mismatch! x:{x.shape}, x_hat:{x_hat.shape}, features:{self.feature_idx}"

        # Loss for the point features (x, y, z)
        loss_shape = self._compute_loss(x[:,:,:3], x_hat[:,:,:3], 
                                        self.recon_mode[0], **kwargs)

        # Loss for the metrics (DTI)
        if x.shape[-1]>3 and x_hat.shape[-1]>3:
            loss_metric = self._compute_loss(x[:,:,3:], x_hat[:,:,3:], 
                                             self.recon_mode[1], **kwargs)
        else:
            loss_metric=0

        # Weight loss
        loss = self.recon_weight[0] * loss_shape + self.recon_weight[1] * loss_metric
        loss_dict = {'loss_recon' : loss.item(), 
                     'loss_recon_shape' : loss_shape.item(),
                     'loss_recon_metric' : loss_metric.item()}
        return loss, loss_dict

    def _compute_loss(self, x, x_hat, mode="likelihood", **kwargs):
        if mode == 'mse':
            return self.mse_loss(x, x_hat, **kwargs)
        elif mode == 'chamfer':
            return self.chamfer_loss(x, x_hat, **kwargs)
        else:
            return self.gaussian_likelihood_loss(x, x_hat, **kwargs)

    @staticmethod
    def gaussian_likelihood_loss(x, x_hat, log_scale=None, **kwargs):
        '''
            Compute reconstruction loss, log p(x|z), log Gaussian likelihood
            Maximize this term (-) which is the probability of input under P(x|z)
        '''
        log_scale = nn.Parameter(torch.Tensor([0.0])) if log_scale is None else log_scale
        scale = torch.exp(log_scale)
        dist = torch.distributions.Normal(x_hat, scale)
        log_pxz = dist.log_prob(x)
        # loss_recon = -log_pxz.sum(dim=(1, 2))
        # return loss_recon.mean()
        return -log_pxz.mean()
        
    @staticmethod
    def mse_loss(x, x_hat, **kwargs):
        return F.mse_loss(x_hat, x)

    @staticmethod
    def chamfer_loss(x, x_hat, bidirectional=True, **kwargs):
        chamferDist = ChamferDistance()
        dist = chamferDist(x, x_hat, bidirectional=bidirectional)
        return 0.5*dist if bidirectional is True else dist


class KLDLoss(BaseLoss):
    def __init__(self, **kwargs):
        pass

    def __call__(self, z, mu, std):
        loss = self.kld_loss(z, mu, std)
        loss_dict = {'loss_kl' : loss.item()}
        return loss, loss_dict
    
    @staticmethod
    def kld_loss(z, mu, std):
        '''
            Compute KL divergence loss (regularization)
            Minimize this term (+) and encourage q to be close to p
        '''
        
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), 
                                       torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
    
        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
    
        kl = (log_qzx - log_pz)  # we want q to be close to p
        # return kl.sum(-1).mean()
        return kl.mean()


class VAELoss(BaseLoss):
    def __init__(self, weights=[1,1], **kwargs):
        self.weights = weights
        self.recon_loss_fn = BundleReconLoss(**kwargs)
        self.kl_loss_fn = KLDLoss(**kwargs)

    def __call__(self, x, x_hat, z, mu, std, kl_anneal_wt=1, **kwargs):
        # Reconstruction loss
        loss_recon, result_dict = self.recon_loss_fn(x, x_hat, **kwargs)

        # KL loss
        loss_kl, result_dict_kl = self.kl_loss_fn(z, mu, std)
        result_dict.update(result_dict_kl)
        
        # Apply weighting
        kl_wt = self.weights[1] * kl_anneal_wt
        loss = self.weights[0] * loss_recon +  kl_wt * loss_kl

        # Save results
        result_dict['loss'] = loss.item()
        result_dict['kl_wt'] = kl_wt
        return loss, result_dict