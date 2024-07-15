import sys
sys.path.append('../src')

import argparse
import logging

import numpy as np
import pandas as pd

from data.fetcher import DataFetcher
from data.data_util import stratified_sampling
from utils.file_util import load_json, save_json, save_list

'''
    Make data config file (.json) for model fine-tuning, validation and testing,
    Use with StreamlineDatasetMulti and StreamlineMapDatasetMulti dataset
'''

def make_dcount(h5path, subjs):
    '''
        Select subjects from h5 data with streamline count
    '''
    # select subjects to use in training
    d_count = load_json(f"{h5path[:-3]}_count.json")
    # d_count = {k:v for k,v in d_count.items() if k.split('/')[0] in subjs}
    d = {}
    for k, v in d_count.items():
        s = k.split('/')[0]
        if s in subjs:
            d[k] = {}
            d[k]['Count'] = v
    # add h5 path to keys
    new_keys = [f'{h5path}/{k}' for k in d.keys()]
    d = dict(zip(new_keys, list(d.values())))
    return d
    

def run(args):
    SEED=2023

    print(DataFetcher(args.input_h5_path).fetch_subject_names())

    df = pd.read_csv(args.metadata_path)
    df['Subject'] = df['Subject'].astype(str)
    subjs_cn = df.loc[df.DX=='CN'].Subject.values
    subjs_pt = df.loc[df.DX!='CN'].Subject.values
    logging.info(f"Total {len(subjs_cn)} controls, and {len(subjs_pt)} patients.")

    # Split control subjects from given site into train/val/test
    train, test = stratified_sampling(subjs_cn, 
                                      n_sample=args.test_pct,
                                      seed=SEED)
    subjs_train = subjs_cn[train].tolist()
    subjs_test = subjs_cn[test].tolist()

    # Add case subjects
    subjs_test.extend(subjs_pt)

    # Make data config used for fine-tuning
    d_count_train = make_dcount(args.input_h5_path, subjs_train)
    d_count_test = make_dcount(args.input_h5_path, subjs_test)

    # Save to output files
    save_json(d_count_train, f"{args.output_prefix}config_train.json")
    save_json(d_count_test, f"{args.output_prefix}config_test.json")
    save_list(subjs_train, f"{args.output_prefix}subj_train.txt")
    save_list(subjs_test, f"{args.output_prefix}subj_test.txt")

def main():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8', level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_h5_path', '-i', type=str, required=True)
    parser.add_argument('--metadata_path', '-metadata', type=str, default=None)
    parser.add_argument('--test_pct', '-test', default=0.2, type=float)
    parser.add_argument('--output_prefix', '-o', type=str, required=True)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()