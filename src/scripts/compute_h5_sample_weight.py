import sys
sys.path.append('../src')

import argparse
import logging

import h5py

from utils.file_util import save_json

'''
    Compute streamline count and frequency given .h5 dataset.
'''

def run(args):

    d_count = {}
    with h5py.File(args.h5_input_path, 'r') as hf:
        for k1 in hf.keys(): # subj
            for k2 in hf[k1].keys(): # bundle
                k = f"{k1}/{k2}"
                count = hf[k].shape[0]
                logging.debug(f"Processing {k} ({count})")
                d_count[k] = count

    total = sum(d_count.values())
    d_prob = {k: v / total for k, v in d_count.items()}
    logging.info(f"Total {total} streamlines from {len(d_prob)} bundles.")

    json_path = f"{args.h5_input_path[:-3]}_count.json" if args.h5_output_path is None else args.h5_output_path
    save_json(d_count, json_path)
    if args.save_prob:
        json_path = f"{args.h5_input_path[:-3]}_prob.json" if args.h5_output_path is None else args.h5_output_path
        save_json(d_prob, json_path)

def main():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8', level=logging.DEBUG, force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('--h5_input_path', '-i', type=str)
    parser.add_argument('--h5_output_path', '-o', type=str, default=None)
    parser.add_argument('--save_prob', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()