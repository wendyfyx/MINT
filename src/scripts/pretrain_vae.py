import sys
sys.path.append('../src')

import logging
import argparse

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model.lightning_modules import VAELightning
from data.data_modules import StreamlineDataModuleMulti
from utils.general_util import set_seed_pl

'''
    Pretrain VAE model
'''

def train(args):

    # General setup
    SEED=args.seed
    DEVICE = args.device
    DEVICE_NUM=args.device_num
    set_seed_pl(SEED)

    # Data module
    dm = StreamlineDataModuleMulti(config_path=args.config_path,
                                   batch_size=args.batch_size, 
                                   n_sample=args.n_sample, 
                                   n_loader=args.n_loader, 
                                   n_worker=args.n_worker, 
                                   feature_idx=args.feature_idx,
                                   apply_centering=args.centering, 
                                   apply_scaling=args.scaling,
                                   seed=SEED)


    # Lightning module
    n_feature = len(args.feature_idx) if args.feature_idx is not None else 7
    anneal_step = args.n_epoch * args.n_sample * args.n_loader

    # Initialize model
    model_kwargs = {'dim_in':n_feature, 'dim_latent':args.dim_latent,
                    'loss_wt':args.loss_wt, 'recon_wt':args.recon_wt, 'recon_mode':args.recon_mode,
                    'anneal_mode':args.anneal_mode, 'anneal_step':anneal_step, 
                    'anneal_cycle':args.anneal_cycle, 'anneal_ratio':args.anneal_ratio,
                    'learning_rate':args.learning_rate, 'weight_decay':args.weight_decay, 
                    'gradient_clip_val':args.gradient_clip_val, 'gradient_clip_algorithm':args.gradient_clip_algorithm,
                    'encoder_type':args.encoder_type, 'encoder_kernels':args.encoder_kernels,
                    'encoder_dims':args.encoder_dims, 'encoder_block':args.encoder_block, 
                    'decoder_type':args.decoder_type, 'decoder_kernels':args.decoder_kernels,
                    'decoder_dims':args.decoder_dims, 'decoder_block':args.decoder_block, 
                    'norm_layer_type':args.norm_layer_type, 'n_norm_groups':args.n_norm_groups,
                    'norm_first':args.norm_first, 'norm_eps':args.norm_eps}
    model=VAELightning(**model_kwargs)

    # Logger and checkpointing
    checkpoint_callback = ModelCheckpoint(every_n_epochs=5, 
                                          save_top_k=-1,
                                          save_last=True)
    logger = TensorBoardLogger(args.log_folder, name=args.model_name, version=args.model_version,
                               default_hp_metric=False)

    # Trainer
    trainer = pl.Trainer(logger=logger, 
                         log_every_n_steps=args.log_every_n_steps,
                         callbacks=[checkpoint_callback],
                         accelerator=DEVICE, 
                         devices=[DEVICE_NUM] if DEVICE=='gpu' else 'auto', 
                         max_epochs=args.n_epoch,
                         enable_progress_bar=True,
                         deterministic=True)
    trainer.fit(model=model, datamodule=dm, train_dataloaders=None)


def main():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8', level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--device_num', type=int, default=0)

    # data args
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_sample', type=int, default=2)
    parser.add_argument('--n_loader', type=int, default=100)
    parser.add_argument('--n_worker', type=int, default=4)
    parser.add_argument('--feature_idx', nargs="*", default=None, type=int)
    parser.add_argument('--centering', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--scaling', default=True, action=argparse.BooleanOptionalAction)

    # model args
    parser.add_argument('--dim_latent', type=int, required=True)
    parser.add_argument('--model_type', type=str, default='ConvVAE')
    parser.add_argument('--encoder_type', type=str, default='Encoder')
    parser.add_argument('--encoder_kernels', nargs="*", default=None, type=int)
    parser.add_argument('--encoder_dims', nargs="*", default=None, type=int)
    parser.add_argument('--encoder_block', type=str, default='ConvBlock')
    parser.add_argument('--decoder_type', type=str, default='Decoder')
    parser.add_argument('--decoder_kernels', nargs="*", default=None, type=int)
    parser.add_argument('--decoder_dims', nargs="*", default=None, type=int)
    parser.add_argument('--decoder_block', type=str, default='ConvBlock')
    parser.add_argument('--norm_layer_type', type=str, default=None)
    parser.add_argument('--n_norm_groups', type=int, default=1)
    parser.add_argument('--norm_first', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--norm_eps', type=float, default=1e-5)

    # loss args
    parser.add_argument('--loss_wt', nargs="*", default=None, type=float)
    parser.add_argument('--recon_wt', nargs="*", default=None, type=float)
    parser.add_argument('--recon_mode', nargs="*", default=None, type=str)
    parser.add_argument('--anneal_mode', type=str, default='sigmoid')
    parser.add_argument('--anneal_cycle', type=int, default=5)
    parser.add_argument('--anneal_ratio', type=float, default=0.5)

    # training args
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--gradient_clip_val', type=float, default=1)
    parser.add_argument('--gradient_clip_algorithm', type=str, default='norm')
    parser.add_argument('--n_epoch', type=int, default=50)

    # logging args
    parser.add_argument('--log_folder', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='my_model')
    parser.add_argument('--model_version', type=str, default=None)
    parser.add_argument('--log_every_n_steps', type=int, default=5)

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()