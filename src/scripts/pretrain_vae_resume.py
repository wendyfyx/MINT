import sys
sys.path.append('../src')

import logging
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from data.data_modules import StreamlineDataModule
from model.lightning_modules import VAELightning, CVAELightning
from utils.general_util import set_seed_pl

'''
    Resume pretraining VAE model from the last checkpoint 
    Use with pretrain_vae.py
'''

def run(args):
    # General setup
    SEED=args.seed
    DEVICE = args.device
    DEVICE_NUM=args.device_num
    set_seed_pl(SEED)

    log_folder = args.log_folder
    model_name = args.model_name
    model_version = args.model_version
    ckpt_name = args.ckpt_name
    ckpt_folder = f"{log_folder}/{model_name}/{model_version}/checkpoints"
    ckpt_path=f"{ckpt_folder}/{ckpt_name}"

    dm = StreamlineDataModule.load_from_checkpoint(ckpt_path)

    checkpoint = torch.load(ckpt_path)
    n_sample = checkpoint['datamodule_hyper_parameters']['n_sample']
    n_loader = checkpoint['datamodule_hyper_parameters']['n_loader']
    n_step_total = min(args.n_epoch*n_sample*n_loader, args.n_step)
    logging.info(f"Total number of steps {n_step_total} (current step is {checkpoint['global_step']})")
    
    # module = CVAELightning if args.conditional else VAELightning
    if args.n_epoch > checkpoint['epoch'] or n_step_total > checkpoint['global_step']:
        model = VAELightning.load_from_checkpoint(ckpt_path, anneal_step=n_step_total)
    else:
        model = VAELightning.load_from_checkpoint(ckpt_path)
    

    logger = TensorBoardLogger(log_folder, model_name, version=model_version, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder,
                                          every_n_epochs=5, save_top_k=-1, save_last=True)
    trainer = pl.Trainer(logger=logger, 
                         log_every_n_steps=args.log_every_n_steps,
                         callbacks=[checkpoint_callback],
                         accelerator=DEVICE, 
                         devices=[DEVICE_NUM] if DEVICE=='gpu' else 'auto', 
                         max_epochs=args.n_epoch,
                         max_steps=n_step_total,
                         deterministic=True)
    trainer.fit(model=model, datamodule=dm, train_dataloaders=None, ckpt_path=ckpt_path)


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

    parser.add_argument('--log_folder', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='vae_model')
    parser.add_argument('--model_version', type=str, default=None)
    parser.add_argument('--ckpt_name', type=str, required=True)

    parser.add_argument('--n_step', type=int, default=30000)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--log_every_n_steps', type=int, default=5)

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()