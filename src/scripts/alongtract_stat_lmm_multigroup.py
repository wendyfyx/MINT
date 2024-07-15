import sys
sys.path.append('/ifs/loni/faculty/thompson/four_d/wfeng/240205-MINT/src')

import argparse
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from evaluation.statistics import multiple_comparisons
from utils.general_util import map_list_with_dict

import warnings
warnings.filterwarnings("ignore")

class concatAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, ' '.join(values))

def run(args):

    # Setting for linear mixed model
    ref_group='MCI'
    save_group=['AD']
    dx_col = f"C(Dx, Treatment(reference='{ref_group}'))"
    vc = {"Subject": "0 + C(Subject)", "Cohort": "0 + C(Cohort)"}
    criteria = f"Metric ~ {dx_col} + Age + Sex"

    # Load metadata
    df_meta = pd.read_csv(args.metadata_path)

    # Load evaluation metrics file
    df = pd.read_csv(args.input_fpath)
    df = pd.merge(df_meta, df, on='Subject').dropna()
    print(df.shape)
    df.Segment = df.Segment.astype(int)
    bundles_all = sorted(df.Bundle.unique())
    
    df_lmm = []
    # Iterate through each bundle 
    for b in bundles_all:
        df_tmp2 = df.loc[(df.Bundle==b)]
        logging.info(f"Running LMM for bundle={b}")

        # Iterate through each segment
        for i in range(1,101):
            df_test = df_tmp2.loc[(df_tmp2.Segment==i)]
            if len(df_test['Subject'].unique()) < args.min_subjects: # don't run stats on less than X subjects
                df_lmm.append([b, i, 1, 0, 0])
            else:
                mdf = sm.MixedLM.from_formula(criteria, groups=np.ones(df_test.shape[0]), 
                                            vc_formula=vc, data=df_test)
                result = mdf.fit()
                # pval = result.pvalues[f"{dx_col}[T.MCI]"]
                # beta = result.params[args.save_col]
                # se = result.bse_fe[args.save_col]
                for g in save_group:
                    df_lmm.append([b, g, i,
                                result.pvalues[f"{dx_col}[T.{g}]"], 
                                result.params[f"{dx_col}[T.{g}]"], 
                                result.bse_fe[f"{dx_col}[T.{g}]"]])
                # df_lmm.append([b, 'AD', i,
                #                result.pvalues[f"{dx_col}[T.AD]"], 
                #                result.params[f"{dx_col}[T.AD]"], 
                #                result.bse_fe[f"{dx_col}[T.AD]"]])
                    
    df_lmm = pd.DataFrame(df_lmm, columns=['Bundle','Group','Segment','p','beta','se'])

    # Save to file
    logging.info(f'Saving result to {args.output_fpath}.')
    df_lmm.to_csv(f"{args.output_fpath}", index=False)



def main():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8', level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_fpath', '-i', type=str, 
                        required=True)
    parser.add_argument('--metadata_path', '-metadata', type=str, 
                        required=True)
    parser.add_argument('--min_subjects', '-min', type=int, 
                        default=10)
    parser.add_argument('--output_fpath', '-o', type=str, 
                        required=True)

    args = parser.parse_args()
    print(args)
    run(args)

if __name__ == '__main__':
    main()