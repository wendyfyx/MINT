import logging
import torch
import lightning.pytorch as pl

from model.anneal_schedule import get_anneal_schedule
from model.loss import VAELoss
from model.model_helper import init_new_decoder, init_new_model, init_model


class VAEModuleBase(pl.LightningModule):
    def __init__(self, dim_in=2, dim_latent=64, 
                 loss_wt=[1,1], recon_wt=[1,1], recon_mode=['likelihood', 'likelihood'], 
                 anneal_mode='', anneal_step=50000, anneal_cycle=1, anneal_ratio=0.5,
                 learning_rate=5e-4, weight_decay=1e-3, 
                 gradient_clip_val=2, gradient_clip_algorithm='norm', **kwargs):
        super().__init__()

        self.dim_in = dim_in
        self.dim_latent = dim_latent
        
        # Optimizer params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # Initialize loss function
        self.anneal_schedule = get_anneal_schedule(anneal_mode, anneal_step, 
                                                   anneal_cycle, anneal_ratio)
        self.loss_fn = VAELoss(weights=loss_wt, recon_mode=recon_mode, recon_weight=recon_wt)
        
        # Initialize model
        self.init_model()
        
        # self.save_hyperparameters()
        
    def init_model(self):
        raise NotImplementedError("Implement model initialization")

    def forward(self, x, y):
        '''Return reconstruction and embeddings'''
        x_hat, z, _, _ = self.model.forward(x, y)
        return x_hat, z
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.learning_rate, 
                                     weight_decay=self.weight_decay)
        return optimizer

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm
        )

    def training_step(self, batch, batch_idx):

        # Forward pass
        x, y = batch
        # print(next(self.model.parameters()).device)
        
        # loss, result_dict = self.model.loss(x, y=y, kl_anneal_wt=self.anneal_schedule[self.global_step-1])
        x_hat, z, mu, std = self.model.forward(x, y)
        loss, result_dict = self.loss_fn(x, x_hat.float(), z, mu, std, 
                                         kl_anneal_wt=self.anneal_schedule[self.global_step-1], 
                                         log_scale=self.model.log_scale)

        # Logging
        new_keys = [f'train/{k}' for k in result_dict.keys()]
        new_data = dict(zip(new_keys, list(result_dict.values())))
        self.log_dict(new_data)
        
        return loss



class VAELightning(VAEModuleBase):
    def __init__(self, **kwargs):
        
        # VAE args 
        self.model_type='VAE'
        self.model_kwargs=kwargs

        self.save_hyperparameters()
        super().__init__(**self.model_kwargs)
        
    def init_model(self):
        self.model = init_new_model(model_type=self.model_type, **self.model_kwargs)
        logging.debug(self.model)


class CVAELightning(VAEModuleBase):
    def __init__(self, ncat=2, dim_label_embed=1, 
                 pretrain_ckpt_path=None, **kwargs):
        
        # VAE args
        self.model_type='CVAE'
        self.ncat = ncat
        self.dim_label_embed = dim_label_embed
        self.pretrain_ckpt_path = pretrain_ckpt_path
        self.model_kwargs=kwargs

        self.save_hyperparameters()
        super().__init__(**self.model_kwargs)
        
    def init_model(self):
        if self.pretrain_ckpt_path is not None:
            model_pretrained = VAELightning.load_from_checkpoint(self.pretrain_ckpt_path)
            logging.info(f'Load pretrained VAE model from {self.pretrain_ckpt_path}')
            encoder = model_pretrained.model.encoder
            dim_latent = self.dim_latent+self.dim_label_embed
            self.model_kwargs['dim_latent'] = dim_latent
            decoder = init_new_decoder(**self.model_kwargs)
            self.model = init_model(self.model_type, dim_latent, encoder, decoder, pretrained=True,
                                   mu=model_pretrained.model.mu, logvar=model_pretrained.model.logvar, 
                                   log_scale=model_pretrained.model.log_scale)
        else:
            self.model = init_new_model(model_type=self.model_type, **self.model_kwargs)
        logging.debug(self.model)