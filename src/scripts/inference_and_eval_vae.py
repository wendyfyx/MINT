import sys
sys.path.append('../src')

import argparse
import logging

import numpy as np
import torch
import lightning.pytorch as pl

from core.AtlasData import AtlasData
from core.BundleVisualizer import get_segment_idx
from core.BundleFeature import int_to_feature
from data.fetcher import DataFetcher
from data.dataset import process_json_config
from model.lightning_modules import VAELightning
from model.model_inference import run_inference_vae, run_decode_vae
from evaluation.reconstruction_metrics import compute_metric, get_segment_metric
from utils.general_util import set_seed_pl
from utils.file_util import save_attr_to_h5, save_dict_to_h5, load_json, key_in_h5

'''
    Inference + along-tract anomaly score
    Default: saves embeddings (no reconstruction) and along-tract anomaly scores
'''

def compute_recon_metric(x, x_recon, feature_idx, bname, segment_idx, k):

    # Get the (x,y,z) and DTI feature indices
    shape_idx = [0,1,2]
    dti_idx = list(set(feature_idx) - set(shape_idx))
    logging.debug(f'DTI features include {[int_to_feature(i) for i in dti_idx]}')

    metrics = ['mae','mape','mean_orig','mean_recon']
    dict_metric = {}

    for m in metrics:
        # whole bundle - all features
        dict_metric[f'{k}/{m}'] = float(compute_metric(x, x_recon, feature_idx=feature_idx, metric=m))

        # segment - all features
        dict_metric[f'{k}/{m}_seg'] = get_segment_metric(x, x_recon, bname, segment_idx=segment_idx, 
                                                         feature_idx=feature_idx, metric=m)

        # whole bundle - shape
        dict_metric[f'{k}/{m}_xyz'] = float(compute_metric(x, x_recon, feature_idx=shape_idx, metric=m))

        # segment - shape
        dict_metric[f'{k}/{m}_seg_xyz'] = get_segment_metric(x, x_recon, bname, segment_idx=segment_idx, 
                                                             feature_idx=shape_idx, metric=m)
        for i in dti_idx:
            dti_feature = int_to_feature(i).lower()
            # whole bundle - DTI
            dict_metric[f'{k}/{m}_{dti_feature}'] = float(compute_metric(x, x_recon, feature_idx=[i], metric=m))

            # segment - DTI
            dict_metric[f'{k}/{m}_seg_{dti_feature}'] = get_segment_metric(x, x_recon, bname, segment_idx=segment_idx, 
                                                                           feature_idx=[i], metric=m)
    
    return dict_metric

def run(args):

    # Set up
    SEED=args.seed
    set_seed_pl(SEED)

    DEVICE=args.device
    if DEVICE == 'cuda':
        DEVICE = f"{args.device}:{args.device_num}"
        # torch.cuda.set_device(args.device_num)

    # Load model and hparams for inference 
    model_ckpt = args.model_ckpt
    model = VAELightning.load_from_checkpoint(model_ckpt, map_location=DEVICE)

    checkpoint = torch.load(model_ckpt, map_location=DEVICE)
    feature_idx = checkpoint['datamodule_hyper_parameters']['feature_idx']

    # Save metadata
    metadata ={'model_ckpt' : model_ckpt,
               'epoch' : checkpoint['epoch'], 
               'global_step' : checkpoint['global_step']}
    if args.output_recon_path is not None:
        save_attr_to_h5(metadata, args.output_recon_path, overwrite=args.overwrite)
    if args.output_z_path is not None:
        save_attr_to_h5(metadata, args.output_z_path, overwrite=args.overwrite)

    # Load config file for inference
    d_config = load_json(args.config_path)
    dim_in = model.dim_in
    dim_latent = model.dim_latent

    # Load atlas data for creating segment indices
    atlas = AtlasData()
    atlas_bundles = {}

    for fpath, k in process_json_config(d_config).items():
        fetcher = DataFetcher(fpath)
        for subj, bundles in k.items():
            for b in bundles:
                k = f'{subj}/{b}'

                #If overwrite is False, and already in file, skip inference
                skip_inference = False
                if args.output_recon_path is not None:
                    skip_inference = key_in_h5(args.output_recon_path, k) # skip if key is found
                if args.output_z_path is not None:
                    skip_inference = key_in_h5(args.output_z_path, k) # skip if key is found
                if args.output_eval_path is not None:
                    skip_inference = key_in_h5(args.output_eval_path, k) # skip if key is found
                if skip_inference:
                    logging.info(f'Skip inference for {fpath}: {k}.')
                    continue

                # Model inference
                x = fetcher.fetch(subj, b)
                if x is None:
                    continue
                x_recon, z, _, _, _, _ = run_inference_vae(model, x, checkpoint=model_ckpt, device=DEVICE,
                                                           split_batch_size=4096, seed=SEED)
                
                # Save reconstruction data to .h5 file
                if args.output_recon_path is not None:
                    dict_recon = {k : x_recon}
                    save_dict_to_h5(dict_recon, args.output_recon_path, 
                                    overwrite=args.overwrite, 
                                    chunks=(min(len(x_recon), 128), 128, dim_in))
                
                # Save embedding z to .h5 file
                if args.output_z_path is not None:
                    dict_z = {k : z}
                    save_dict_to_h5(dict_z, args.output_z_path, 
                                    overwrite=args.overwrite, 
                                    chunks=(min((len(z)), 1024), dim_latent))

                # Save along-tract metric to .h5 file
                if args.output_eval_path is not None:
                    # Retrieve atlas bundle and compute segment
                    if b in atlas_bundles:
                        ref_bundle = atlas_bundles[b]
                    else:
                        ref_bundle = atlas.fetch_bundle(b)
                        atlas_bundles[b] = ref_bundle
                    segment_idx = get_segment_idx(x, ref_bundle, n_segments=args.n_segments)

                    # Compute along tract metrics and save
                    dict_metric = compute_recon_metric(x, x_recon, feature_idx, b, segment_idx, k)
                    
                    # Save anomaly score to .h5
                    save_dict_to_h5(dict_metric, args.output_eval_path, overwrite=args.overwrite)


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
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--n_segments', type=int, default=100)

    # output args
    parser.add_argument('--output_z_path', '-oz', type=str, default=None)
    parser.add_argument('--output_recon_path', '-or', type=str, default=None)
    parser.add_argument('--output_eval_path', '-om', type=str, default=None)
    parser.add_argument('--overwrite', '-w', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()