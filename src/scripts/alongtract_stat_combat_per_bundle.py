import sys
sys.path.append('../src')

import os
import argparse
import logging

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer

from evaluation.combat_harmonization import run_combat, apply_combat

'''
    Train and apply ComBat for each bundle.
'''

def transform_df(df):
    '''
        Convert dataframe where with subject as rows, and bundle|segment as columns
    '''
    df = df.astype({'Segment':str})
    df = df.pivot(index='Subject', columns=['Bundle','Segment'], values='Metric')
    df.columns = df.columns.map('|'.join)
    return df


def inverse_transform_df(df):
    '''
        Convert dataframe back to the original "melted" format
    '''
    df = df.reset_index().melt(id_vars=['Subject'], 
                               var_name='var', value_name='Metric') 
    df[['Bundle', 'Segment']] = df['var'].str.split('|', n=1, expand=True)
    df = df.drop(columns=['var']).astype({'Segment':int}).sort_values(by=['Subject','Segment'])
    df = df[df['Metric'].notna()]
    return df


def transform_df_bundle(df):
    '''
        Convert dataframe where with subject and bundles as rows, and segments as columns
    '''
    return df.pivot(index=['Subject','Bundle'], columns='Segment', values='Metric').reset_index()


def inverse_transform_df_bundle(df):
    '''
        Convert dataframe back to the original "melted" format
    '''
    return df.melt(id_vars=['Subject','Bundle'], value_name='Metric')


def impute_data(df):
    df = transform_df(df)
    df_imputed = df.copy()
    
    imputer = KNNImputer(n_neighbors=3)
    data = imputer.fit_transform(df.values)
    df_imputed[:] = data
    return inverse_transform_df(df_imputed)


def run(args):
    if not os.path.exists(f'{args.output_folder}/combat'):
        os.makedirs(f'{args.output_folder}/combat')

    # Read in metadata
    df_meta = pd.read_csv(args.metadata_path)

    # Read in along-tract metrics for feature
    feature_suffix = "" if args.feature=='all' else f"_{args.feature.lower()}"
    df_train = pd.read_csv(f"{args.input_folder}/eval_train_metrics_{args.metric.lower()}_seg{feature_suffix}.csv")
    df_test = pd.read_csv(f"{args.input_folder}/eval_test_metrics_{args.metric.lower()}_seg{feature_suffix}.csv")

    # Impute training data
    df_train = impute_data(df_train)

    # Fit ComBat on training data
    df_train = transform_df_bundle(df_train)
    for b in df_train.Bundle.unique():
        df_b = df_train.loc[df_train.Bundle==b].drop(columns='Bundle').set_index('Subject')
        df_covar = pd.merge(df_b.reset_index(), df_meta, on='Subject')[['Subject','Age','Sex','Protocol']]
        combat_fpath = f'{args.output_folder}/combat/eval_train_combat_{args.metric.lower()}_seg{feature_suffix}_{b}.pkl'
        _ = run_combat(df_b.values, df_covar, combat_fpath=f"{combat_fpath}", do_impute=False)

    # Apply ComBat on test data
    df_test = transform_df_bundle(df_test)
    segment_cols = [i+1 for i in range(100)]
    df_corr = df_test.copy()
    for b in df_test.Bundle.unique():
        df_b = df_test.loc[df_test.Bundle==b].drop(columns='Bundle').set_index('Subject')
        df_covar = pd.merge(df_b.reset_index(), df_meta, on='Subject')[['Subject','Age','Sex','Protocol']]
        
        combat_fpath = f'{args.output_folder}/combat/eval_train_combat_{args.metric.lower()}_seg{feature_suffix}_{b}.pkl'
        data_corr = apply_combat(df_b.values, df_covar.Protocol, combat_fpath)
        df_corr.loc[df_corr.Bundle==b, segment_cols] = data_corr
    df_corr = inverse_transform_df_bundle(df_corr)
    
    # Save output
    output_fpath=f'{args.output_folder}/eval_test_metrics_{args.metric.lower()}_seg{feature_suffix}_combat_perbundle.csv'
    df_corr.to_csv(output_fpath, index=False)
    logging.info(f"Saved ComBat corrected data to {output_fpath}.")

def main():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8', level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', '-i', type=str, 
                        required=True)
    parser.add_argument('--metadata_path', '-metadata', type=str, 
                        required=True)
    parser.add_argument('--metric', '-metric', type=str, 
                        default='mae')
    parser.add_argument('--feature', '-feature', type=str, 
                        default='fa')
    # parser.add_argument('--features', '-features', nargs="+", type=str, 
    #                     default=['xyz', 'fa', 'md', 'rd', 'axd'])
    parser.add_argument('--output_folder', '-o', type=str, 
                        required=True)

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()