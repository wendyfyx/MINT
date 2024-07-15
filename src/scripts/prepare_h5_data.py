import sys
sys.path.append('../src')

import logging
import argparse

import numpy as np
import pandas as pd

from core.BundleName import Atlas30
from core.BundleFeature import BundleFeature
from data.fetcher import DataFetcher

SEED=2024

class MyDataFetcher(DataFetcher):
    def __init__(self, h5_data_path, input_data_path, 
                 split=None, subjects=None, bundles=None, 
                 rec_bundles_folder='rec_bundles/',
                 org_bundles_folder='org_bundles/',
                 dti_folder='anatomical_measures/',
                 **kwargs):
        
        # folder with MNI space bundles
        self.rec_bundles_folder = rec_bundles_folder

        # folder with native space bundles
        self.org_bundles_folder = org_bundles_folder

        # folder with DTI measures (must be in the same space as org_bundles)
        self.dti_folder = dti_folder
        
        super().__init__(h5_data_path, input_data_path, split=split, 
                         subjects=subjects, bundles=bundles, **kwargs)

    def set_args(self):
        self.kwargs.setdefault('rec_bundles_folder', self.rec_bundles_folder) 
        self.kwargs.setdefault('org_bundles_folder', self.org_bundles_folder) 
        self.kwargs.setdefault('dti_folder', self.dti_folder) 

        # Custom function for parsing bundle name from file name (override as needed)
        self.kwargs.setdefault('parse_tract_func', self.parse_tract_name)

        self.kwargs.setdefault('data_folder', self.data_folder)
        if self.bundles is not None:
            self.kwargs.setdefault('tracts_include', self.bundles)


def run(args):

    n_lines_per_chunk = 96 # h5 file chunk size
    input_data_path="../example_data/subjects_small"

    bundles = [i.name for i in Atlas30] if args.bundles is None else args.bundles
    dti_features = [i.filename for i in BundleFeature] if args.features is None else args.features
    logging.info(f"Load {len(bundles)} bundles {bundles} and {len(dti_features)} features {dti_features}.")

    # min_lines must be equal or larger than chunk size
    kwargs = {'n_points' : 128, 'min_lines' : n_lines_per_chunk, 
              'preprocess' : '3d', 'normalize' : False,
              'save_ptct' : False, 'rng' : np.random.default_rng(SEED),
              'dti': dti_features}
    
    fetcher=MyDataFetcher(h5_data_path=args.output_path, 
                          input_data_path=args.input_path,
                          subjects=args.subjects, 
                          bundles=bundles, 
                          split=args.split, **kwargs)
    fetcher.create_dataset(mode=args.h5_mode, n_lines_per_chunk=n_lines_per_chunk)


def main():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s [%(pathname)s %(funcName)s %(lineno)d]',
                        datefmt='%H:%M:%S',
                        encoding='utf-8', level=logging.INFO, force=True)

    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument('--input_path', '-i', 
                        type=str, required=True)
    parser.add_argument('--output_path', '-o', 
                        type=str, default='./data.h5')
    parser.add_argument('--subjects', '-subjects', nargs="+", 
                        type=str, default='all')
    parser.add_argument('--bundles', '-bundles', nargs="+", 
                        type=str, default=None)
    parser.add_argument('--features', '-features', nargs="+", 
                        type=str, default=None)
    parser.add_argument('--split', '-split',
                        type=str, default=None)
    parser.add_argument('--rec_bundles_folder', '-rec_bundles_folder',
                        type=str, default='rec_bundles/')
    parser.add_argument('--org_bundles_folder', '-org_bundles_folder',
                        type=str, default='org_bundles/')
    parser.add_argument('--dti_folder', '-dti_folder',
                        type=str, default='anatomical_measures/')
    parser.add_argument('--h5_mode', '-mode', 
                        type=str, default='w')

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()