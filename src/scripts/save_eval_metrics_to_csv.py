import sys
sys.path.append('/ifs/loni/faculty/thompson/four_d/wfeng/240205-MINT/src')

import argparse
import logging

import h5py
import pandas as pd

from core.BundleFeature import BundleFeature

def save_metrics(h5_path, output_prefix, item_keys=['mae','mae_xyz','mae_fa']):
    ls = []
    columns = ['Subject','Bundle'] + [i.upper() for i in item_keys]
    fname = f"{output_prefix}_metrics.csv"
    logging.info(f'Saving {len(item_keys)} metrics {item_keys} to {fname}.')
    with h5py.File(h5_path, 'r') as hf:
        for subj in hf.keys():
            for bundle in hf[subj].keys():
                data = [subj, bundle]
                for k in item_keys:
                    data.append(float(hf[f"{subj}/{bundle}/{k}"][...]))
                ls.append(data)
    df_mae=pd.DataFrame(ls, columns=columns)
    df_mae.to_csv(fname, index=False)


def save_along_tract_metrics(h5_path, output_prefix, metric='mae_seg'):
    segment_cols = [str(i+1) for i in range(100)]
    fname = fname = f"{output_prefix}_metrics_{metric}.csv"
    logging.info(f'Saving metrics {metric} to {fname}.')
    with h5py.File(h5_path, 'r') as hf:
        for i, subj in enumerate(hf.keys()):
            ls = []
            for bundle in hf[subj].keys():
                k = f"{subj}/{bundle}"
                data =[subj, bundle]
                data.extend(hf[f"{k}/{metric}"][...])
                ls.append(data)
            df_mae=pd.DataFrame(ls, columns=['Subject','Bundle'] + segment_cols)
            df_mae=df_mae.melt(id_vars=['Subject','Bundle'], value_vars=segment_cols, 
                           var_name='Segment', value_name='Metric')
            save_header = True if i == 0 else False
            mode = 'w' if i == 0 else 'a'
            df_mae.to_csv(fname, index=False, header=save_header, mode=mode)

def run(args):
    save_metrics(args.h5_path, args.output_prefix, item_keys=args.keys)
    for k in args.keys_seg:
        save_along_tract_metrics(args.h5_path, args.output_prefix, metric=k)

def main():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8', level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()

    features = ['xyz'] + [i.name.lower() for i in BundleFeature]
    metrics = ['mae','mean_orig','mean_recon']
    keys = metrics + [f"{m}_{f}" for m in metrics for f in features]
    keys_seg = [f"{m}_seg" for m in metrics] + [f"{m}_seg_{f}" for m in metrics for f in features]

    parser.add_argument('--h5_path', '-i', type=str, required=True)
    parser.add_argument('--output_prefix', '-o', type=str, required=True)
    parser.add_argument('--keys', nargs="*", default=keys, type=str)
    parser.add_argument('--keys_seg', nargs="*", default=keys_seg, type=str)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()