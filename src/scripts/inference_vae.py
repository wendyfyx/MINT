import sys
sys.path.append('../src')

import argparse
import logging

import torch
import lightning.pytorch as pl

from data.fetcher import DataFetcher
from data.dataset import process_json_config
from model.lightning_modules import VAELightning
from model.model_inference import run_inference_vae
from utils.general_util import set_seed_pl
from utils.file_util import save_dict_to_h5, load_json

'''
    Inference script for saving both reconstruction and embeddings 
    using config json file (support reading from multiple .h5 files)
'''

def run(args):

    # Set up
    SEED=args.seed
    DEVICE = args.device
    set_seed_pl(SEED)
    if DEVICE == 'cuda':
        torch.cuda.set_device(args.device_num)

    # Load model and hparams for inference 
    model_ckpt = args.model_ckpt
    model = VAELightning.load_from_checkpoint(model_ckpt)

    # h5_output_recon_path = f"{args.output_folder_path}/x_recon.h5"
    # h5_output_z_path = f"{args.output_folder_path}/z.h5"

    # Save inference data
    d_config = load_json(args.config_path)
    dim_in = model.dim_in
    dim_latent = model.dim_latent

    for fpath, k in process_json_config(d_config).items():
        fetcher = DataFetcher(fpath)
        for subj, bundles in k.items():
            for b in bundles:

                # Model inference
                bundle = fetcher.fetch(subj, b)
                x_recon, z, _, _ = run_inference_vae(model, bundle, 
                                                     checkpoint=model_ckpt,
                                                     device=DEVICE)

                # Save to .h5 file
                if args.output_recon_path is not None:
                    dict_recon = {f'{subj}/{b}' : x_recon}
                    chunk_len_recon = min(len(x_recon), 128)
                    save_dict_to_h5(dict_recon, args.output_recon_path, 
                                    overwrite=args.overwrite, chunks=(chunk_len_recon, 128, dim_in))
                if args.output_z_path is not None:
                    dict_z = {f'{subj}/{b}' : z}
                    chunk_len_z = min((len(z)), 1024)
                    save_dict_to_h5(dict_z, args.output_z_path, 
                                    overwrite=args.overwrite, chunks=(chunk_len_z, dim_latent))


def main():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8', level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--device_num', type=int, default=0)

    # inference args
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--model_ckpt', '-m', type=str)
    parser.add_argument('--output_z_path', '-oz', type=str, default=None)
    parser.add_argument('--output_recon_path', '-or', type=str, default=None)
    parser.add_argument('--overwrite', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()