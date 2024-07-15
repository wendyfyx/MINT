import importlib
import logging
import torch.nn as nn


def init_weights(m):
    '''Initialize model weight with Xavier uniform'''
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias,0)


def init_new_encoder(dim_in=7, encoder_type='Encoder', encoder_kernels=[63,31,15], 
                     encoder_dims=[16,32,64], encoder_block='ConvBlock',
                     norm_layer_type='bn', n_norm_groups=1, norm_first=False, norm_eps=1e-5, **kwargs):
        '''Initialize new encoder'''
        encoder_cls = getattr(importlib.import_module("model.arch.encoder_decoder"), encoder_type)
        encoder_block_cls = getattr(importlib.import_module("model.arch.modules"), encoder_block)
        model_args = {'line_len' : 128, 'print_layer' : False,
                      'norm_layer_type' : norm_layer_type, 'n_norm_groups' : n_norm_groups,
                      'norm_first' : norm_first, 'norm_eps' : norm_eps}
        encoder = encoder_cls(dim_in, block=encoder_block_cls, 
                              kernel_dims=encoder_kernels, layer_dims=encoder_dims, **model_args)
        init_weights(encoder)
        logging.info(f'Initialized new {encoder_type} with {encoder_block}.')
        return encoder


def init_new_decoder(dim_latent=2, dim_in=7, 
                     decoder_type='Decoder', decoder_kernels=[65,31,63], 
                     decoder_dims=[64,32,16], decoder_block='ConvBlock',
                     norm_layer_type='bn', n_norm_groups=1, norm_first=False, norm_eps=1e-5, **kwargs):
        '''Initialize new decoder'''
        decoder_cls = getattr(importlib.import_module("model.arch.encoder_decoder"), decoder_type)
        decoder_block_cls = getattr(importlib.import_module("model.arch.modules"), decoder_block)
        model_args = {'line_len' : 128, 'print_layer' : False,
                      'norm_layer_type' : norm_layer_type, 'n_norm_groups' : n_norm_groups,
                      'norm_first' : norm_first, 'norm_eps' : norm_eps}
        decoder = decoder_cls(dim_latent, dim_in, block=decoder_block_cls, 
                              kernel_dims=decoder_kernels, layer_dims=decoder_dims, **model_args)
        init_weights(decoder)
        logging.info(f'Initialized new {decoder_type} with {decoder_block}.')
        return decoder


def init_new_model(dim_in=7, dim_latent=2, model_type='VAE', **model_kwargs):
    '''Initialize new model with specified encoder and decoder parameters'''
    encoder = init_new_encoder(dim_in, **model_kwargs)
    decoder = init_new_decoder(dim_latent, dim_in, **model_kwargs)
    return init_model(model_type, dim_latent, encoder, decoder, pretrained=False)


def init_model(model_type, dim_latent, encoder, decoder, pretrained=False, **kwargs):
    '''Initialize model'''
    model_cls = getattr(importlib.import_module(f"model.arch.{model_type.lower()}"), model_type)
    model = model_cls(dim_latent, encoder, decoder, **kwargs)
    if not pretrained: # only init mu and logvar weights when initializing new model
        init_weights(model.mu)
        init_weights(model.logvar)
    if 'cvae' in model_type.lower(): # initialize the new label embedding layers
        init_weights(model.label_embeddings)
    logging.info(f'Initialized {model_type} model.')
    return model.double()
