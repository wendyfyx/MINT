import sys
sys.path.append('../src')

import logging
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from data.data_modules import StreamlineDataModuleMulti
from model.lightning_modules import VAELightning
from utils.general_util import set_seed_pl
from utils.dataset_helper import get_subjects_stratified

'''
    Fine-tuning with StreamlineBootstrapDataset (iterable-style) 
    with metadata support. Use this for large dataset.
'''

GLOBAL_SEED=2023

def run(args):
    # General setup
    SEED=args.seed
    DEVICE = args.device
    DEVICE_NUM=args.device_num
    set_seed_pl(SEED)

    # Load data modules from checkpoint
    ckpt_path = args.pretrain_ckpt_path
    checkpoint = torch.load(ckpt_path)
    dm_params = checkpoint['datamodule_hyper_parameters']

    dm = StreamlineDataModuleMulti(config_path=args.config_path, 
                                   batch_size=args.batch_size if args.batch_size is not None else dm_params['batch_size'],
                                   n_sample=dm_params['n_sample'],
                                   n_loader=dm_params['n_loader'],
                                   n_worker=args.n_worker if args.n_worker is not None else dm_params['n_worker'],
                                   feature_idx=dm_params['feature_idx'], 
                                   apply_centering=dm_params['apply_centering'],
                                   apply_scaling=dm_params['apply_scaling'], 
                                   seed=SEED)
                                   
    # Load model modules form checkpoint
    n_step_total = checkpoint['global_step'] + args.n_step
    model = VAELightning.load_from_checkpoint(ckpt_path, anneal_mode='', anneal_step=n_step_total)
    
    # Set up logger and checkpoint
    finetune_ckpt_folder = f"{args.output_log_folder}/{args.finetune_model_name}/{args.finetune_model_version}/checkpoints"

    logger = TensorBoardLogger(args.output_log_folder, args.finetune_model_name, 
                               version=args.finetune_model_version, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(dirpath=finetune_ckpt_folder,
                                          every_n_train_steps=args.checkpoint_every_n_steps, 
                                          save_top_k=-1, save_last=True)

    # Set up fine-tune trainer
    trainer = pl.Trainer(logger=logger, 
                         log_every_n_steps=args.log_every_n_steps,
                         callbacks=[checkpoint_callback],
                         accelerator=DEVICE, 
                         devices=[DEVICE_NUM] if DEVICE=='gpu' else 'auto', 
                         max_steps=n_step_total,
                         deterministic=True)
    trainer.fit(model=model, datamodule=dm, train_dataloaders=None, ckpt_path=ckpt_path)


def main():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8', level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--device_num', type=int, default=0)

    # Data args
    parser.add_argument('--config_path', type=str, required=True)

    # Model & logging args
    parser.add_argument('--pretrain_ckpt_path', type=str, required=True)
    parser.add_argument('--output_log_folder', type=str, default='.')
    parser.add_argument('--finetune_model_name', type=str, default='finetune_vae')
    parser.add_argument('--finetune_model_version', type=str, default='version0')
    parser.add_argument('--log_every_n_steps', type=int, default=5)
    parser.add_argument('--checkpoint_every_n_steps', type=int, default=1000)

    # Finetune args
    parser.add_argument('--n_step', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=None)

    args = parser.parse_args()
    run(args)
    # print(args)

if __name__ == '__main__':
    main()