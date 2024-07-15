import logging

import h5py
import torch

from model.loss import VAELoss
from data.transform import Normalize
from data.dataset import make_data_loader
from utils.file_util import save_dict_to_h5


def run_inference_vae(model, X, Y=None, checkpoint=None,
                      device='cpu', split_batch_size=2048,
                      seed=0, encode_only=False, default_elbo=False,
                    **kwargs):
    '''Run VAE model inference and return the reconstruction and embeddings'''

    # Load checkpoint if not loaded already
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location=device)
    
    # Get data transform and loss parameters
    apply_centering=checkpoint['datamodule_hyper_parameters']['apply_centering']
    apply_scaling=checkpoint['datamodule_hyper_parameters']['apply_scaling']
    feature_idx=checkpoint['datamodule_hyper_parameters']['feature_idx']

    if default_elbo:
        loss_fn = VAELoss()
    else:
        loss_fn = VAELoss(weights=checkpoint['hyper_parameters']['loss_wt'], 
                        recon_mode=checkpoint['hyper_parameters']['recon_mode'],
                        recon_weight=checkpoint['hyper_parameters']['recon_wt'])
    
    with torch.no_grad():
        model.eval().to(device)
    
        centroid = 'atlas' if apply_centering else None
        radius = 'atlas' if apply_scaling else None
        transform = Normalize(centroid=centroid, radius=radius, feature_idx=feature_idx)
    
        # Convert input to torch tensors
        X = transform(X[..., feature_idx])
        X = torch.Tensor(X).to(device)
        if Y is None:
            Y = torch.Tensor([0]).repeat(len(X)).to(device)

        # Process in batches if data is too big
        if split_batch_size < len(X) and split_batch_size > 0:
            loader = make_data_loader(X, Y, seed=seed, 
                                      batch_size=split_batch_size, 
                                      shuffle=False)
            Z = torch.Tensor([]).to(device) # storing embedding
            X_recon = torch.Tensor([]).to(device) # storing reconstruction
            Mu = torch.Tensor([]).to(device) 
            Std = torch.Tensor([]).to(device) 
            for x, y in loader:
                if encode_only:
                    z, mu, std = model.model.encode_z(x)
                else:
                    x_recon, z, mu, std = model.model(x, y)
                    X_recon = torch.cat((X_recon, x_recon), axis=0)
                Z = torch.cat((Z, z), axis=0)
                Mu = torch.cat((Mu, mu), axis=0)
                Std = torch.cat((Std, std), axis=0)
                torch.cuda.empty_cache()
        else:
            if encode_only:
                Z, Mu, Std = model.model.encode_z(X)
            else:
                X_recon, Z, Mu, Std = model.model(X, Y)

        if not encode_only:
            # Compute loss
            loss, result_dict = loss_fn(X, X_recon, Z, Mu, Std, log_scale=model.model.log_scale)

            # Post-process
            X_recon = X_recon.cpu().detach().numpy()
            X_recon = transform.unnormalize(X_recon)

        Z = Z.cpu().detach().numpy()
        Mu = Mu.cpu().detach().numpy()
        Std = Std.cpu().detach().numpy()

        if encode_only:
            return Z, Mu, Std
        else:
            return X_recon, Z, Mu, Std, loss.item(), result_dict


def run_decode_vae(model, Z, Y=None, checkpoint=None,
                   device='cpu', split_batch_size=2048, 
                   seed=0):
    
    # Load checkpoint if not loaded already
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location=device)
        
    apply_centering=checkpoint['datamodule_hyper_parameters']['apply_centering']
    apply_scaling=checkpoint['datamodule_hyper_parameters']['apply_scaling']
    centroid = 'atlas' if apply_centering else None
    radius = 'atlas' if apply_scaling else None
    feature_idx=checkpoint['datamodule_hyper_parameters']['feature_idx']
    
    with torch.no_grad():
        # Decode embeddings
        Z = torch.Tensor(Z).to(device)
        model.eval().to(device)

        if split_batch_size < len(Z):
            loader = make_data_loader(Z, seed=seed, 
                                      batch_size=split_batch_size, 
                                      shuffle=False)
            out = torch.Tensor([]).to(device) # storing embedding
            for batch in loader:
                x_dec = model.model.decode(batch[0].double())
                out = torch.cat((out, x_dec), axis=0)
        else:
            out = model.model.decode(Z.double())

        # Unnormalize
        out = out.data.cpu().numpy()
        transform = Normalize(centroid=centroid, radius=radius, feature_idx=feature_idx)
        out = transform.unnormalize(out)
        return out


def save_inference_to_h5(h5_path, subj, bundle, data, mode='a', overwrite=False):
    '''Save inference data (dict) for each bundle to h5_path under the key subj/bundle'''
    new_keys = [f'{subj}/{bundle}/{k}' for k in data.keys()]
    new_data = dict(zip(new_keys, list(data.values())))
    save_dict_to_h5(new_data, h5_path, mode=mode, overwrite=overwrite, 
                    compression='gzip')