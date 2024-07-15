import logging
import glob

import pandas as pd
import numpy as np

from dipy.io.streamline import load_tractogram
from dipy.segment.metric import mdf
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines, \
                                     orient_by_streamline, transform_streamlines
from fury.utils import map_coordinates_3d_4d

import nibabel as nib
from nibabel.affines import apply_affine
from nibabel.streamlines.array_sequence import ArraySequence

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from utils.general_util import random_select, random_shuffle
from utils.file_util import load_pickle
from core.BundleName import name_to_int, int_to_name


def load_nib(fpath):
    '''Load .nii files'''
    img = nib.load(fpath)
    return img.get_fdata(), img.affine


def load_streamlines(fpath, n_points=128, subsample=None, subsample_mode='random',
                     align_streamline=None, preprocess='3d', 
                     save_ptct=False, rng=0, **kwargs):

    '''
        Load streamlines from one .trk file (one tract).
        PARAMETERS:
            fpath               : file path of streamline (.trk) files.
            n_points            : number of points per streamline.
            subsample_mode      : mode for subsampling (supports random and farthest line sampling)
            subsample           : percentage of lines to subsample
            preprocess          : can be either 2D or 3D. For ConvVAE, use 3D (n_lines, n_points, 3).
            rng                 : numpy random generator, np.random.rng(seed).
        RETURNS:
            lines               : processed streamline bundle as np.array, return ArraySequence when 
                                  n_points is not provided.
            ptct                : original number of points per streamlines
            subsample_idx       : indices of subsampled streamlines
    '''

    lines = load_tractogram(fpath, reference="same", bbox_valid_check=False).streamlines
    subsample_idx = None

    fname = fpath.split('/')[-1]
    logging.debug(f"Loaded {fname} with {len(lines)} lines each with {n_points} points.")

    # Load all points
    if n_points is not None:
        if isinstance(n_points, int):
            lines = set_number_of_points(lines, n_points)
        elif n_points < 1:
            lines = resample_lines_by_percent(lines, n_points)

    # Subsample from bundle
    if subsample is not None:
        # Use pre-loaded indices
        if isinstance(subsample, np.ndarray):
            subsample_idx = subsample
        # Random sampling
        elif subsample_mode.lower() == 'random':
            subsample_idx = random_sampling(lines, subsample, rng)
            logging.debug(f"Subsampled {len(subsample_idx)} streamlines (random).")
        # Farthest sampling
        elif subsample_mode.lower() == 'farthest': # n_points must be set
            assert n_points is not None, "Must set n_points for farthest sampling."
            subsample_idx = farthest_sampling(lines.get_data().reshape((-1, n_points, 3)), subsample, rng)
            logging.debug(f"Subsampled {len(subsample_idx)} streamlines (farthest).")
        lines = lines[subsample_idx]

    # Align all streamline direction
    if align_streamline is not None:
        lines = orient_by_streamline(lines, align_streamline, n_points=n_points)

    # Note that ptct returned is for all streamlines after resampling and subsampling
    ptct = [len(l) for l in lines] if save_ptct else None

    # Load as point set (N_points, 3)
    if preprocess == "2d":
        lines = lines.get_data()
        logging.debug(f"Preprocessed lines into {preprocess} with shape {lines.shape}")
    # Load as line set (N_lines, N_points, 3)
    elif preprocess == "3d":
        if n_points is None or n_points < 1:
            logging.warning("Cannot process into 3D if n_points=None, returning ArraySequence")
            return lines
        # if subsample is not None:
        #     n_lines = min(n_lines, len(lines))
        # else:
        #     n_lines = len(lines)
        lines = lines.get_data().reshape((-1, n_points, 3))
        logging.debug(f"Preprocessed lines into {preprocess} with shape {lines.shape}")

    return lines, ptct, subsample_idx


def load_bundles(folder_path, parse_tract_func, min_lines=2, 
                align_bundles_path=None, tracts_exclude=None, tracts_include=None,
                sub_folder_path="rec_bundles/", subsample=None, susample_mode='random', **kwargs):
    '''Load bundles in folder, sorted alphabetically by tract name. 
        PARAMETERS:
            folder_path         : the root folder path for each subject is (i.e. Subj01/).
            parse_tract_func    : a custom function that parse the file name to get the 
                                tract name (moved_CST_L__recognized.trk -> CST_L)
            min_lines           : minimum number of lines in a tract. Discard if below this
                                threshold.
            align_bundles_path  : file path of streamlines to align current bundle with
            tracts_exclude      : a list containing tracts to exclude (takes precedence over tracts_include).
            tracts_include      : a list containing tracts to include.
            sub_folder_path     : (OPTIONAL) it's for when the bundle files are nested in other 
                                folders (Subj01/rec_bundles/*.trk).
            subsample           : subsampling method for each bundle. Accept percentage (float), number of 
                                  lines (int) or preloaded indices (dict, string:array)
        RETURNS:
            lines               : streamlines concatenated from all available bundles.
            bundle_idx          : dict, where the key is bundle name, and the value is a tuple 
                                  (starting indexof bundle in lines, length of bundles).
            all_ptct            : list, original number of points per streamlines for all bundles loaded
            bundle_subsample_idx: dict, Indices of subsampled streamlines for all bundles loaded
    '''

    lines = [] # for concatenating streamlines
    bundle_idx = {} # for indexing bundle
    all_ptct = [] # original point count per line
    bundle_subsample_idx = {} # saving streamline indices from subsampling
    logging.debug(f"Loading bundles from {folder_path}...")

    if not folder_path.endswith("/"):
        folder_path = folder_path + "/"
    if sub_folder_path:
        if not sub_folder_path.endswith("/"):
            sub_folder_path = sub_folder_path + "/"

    # Load the centroid streamline from atlas data for aligning streamlines
    if align_bundles_path is not None:
        d_centroid = load_pickle(align_bundles_path)
        logging.debug(f"Loaded bundles to align with from {align_bundles_path}, {d_centroid['AF_L'].shape}")

    lines_count = 0
    for fpath in sorted(list(glob.glob(folder_path + sub_folder_path + "*.trk"))):  
        fname = fpath.split('/')[-1]
        bundle_name = parse_tract_func(fname)
        
        if tracts_exclude: # Tracts to exclude
            if bundle_name in tracts_exclude:
                continue
        if tracts_include: # Tracts to include
            if bundle_name not in tracts_include:
                continue

        align_streamline = d_centroid[bundle_name] if align_bundles_path is not None else None

        # Load streamlines, and specify subsampling arguments
        if isinstance(subsample, dict) and len(subsample)>0:
            subsample_arg = subsample[bundle_name] # use previously saved indices
        else: 
            subsample_arg = subsample
        bundle, ptct, subsample_idx = load_streamlines(fpath, align_streamline=align_streamline, 
                                                       subsample=subsample_arg, susample_mode=susample_mode,
                                                       **kwargs)
        if subsample_idx is not None: # save indices of subsampled streamlines for bundle
            bundle_subsample_idx[bundle_name] = subsample_idx

        # Discard bundle if streamline count below threshold (after subsampling if specified)
        if len(bundle) < min_lines: 
            continue
            
        lines.append(bundle)
        bundle_idx[bundle_name]=[lines_count, len(bundle)]
        if ptct is not None:
            all_ptct.extend(ptct)
        lines_count += len(bundle)

    if len(bundle_subsample_idx) == 0:
        bundle_subsample_idx = None
    if len(lines)==0:
        return np.array([]), bundle_idx, all_ptct, bundle_subsample_idx
    lines = np.concatenate(lines)

    return lines, bundle_idx, all_ptct, bundle_subsample_idx


def make_y(bundle_idx):
    '''
        Make labels from bundle information. Return 1D array of index, and 
        a dictionary of the corresponding bundle names.
    '''
    y = []
    bundle_num = {}
    
    for idx, bundle in enumerate(sorted(bundle_idx)):
        label = name_to_int(bundle)
        y.append([label] * bundle_idx[bundle][1])
        bundle_num[label] = bundle

    if len(y)==0:
        return np.array([]), bundle_num
    y = np.concatenate(y)
    return y, bundle_num


def dti2bundle(lines_org, dtimap, affine, is_2d=False):
    '''
        Maps DTI volume to streamlines and returns DTI metric for each points
        Can support 2D point set and 3D lint set (specify if input is 2D).
    '''
    if is_2d:
        X_native = apply_affine(np.linalg.inv(affine), lines_org)
    else:
        X_native = transform_streamlines(lines_org, np.linalg.inv(affine))
        try: 
            X_native = np.array(X_native)
        except:
            X_native = ArraySequence(X_native).get_data()
    return map_coordinates_3d_4d(dtimap, X_native).T


def normalize_unit_sphere(x, feature_idx=[0,1,2], centroid=None, dist=None, **kwargs):
    '''Normalize x, y, z features of input to unit sphere'''
    x = x.copy()
    x_sel = x[..., feature_idx]
    if centroid is None:
        centroid = np.mean(x_sel, axis=tuple(np.arange(len(x.shape)-1)))
    x_sel -= centroid
    if dist is None:
        dist = np.max(np.sqrt(np.sum(abs(x_sel)**2,axis=-1)))
    x_sel /= dist
    x[..., feature_idx] = x_sel
    logging.debug(f"Applied unit sphere normalization on dim {feature_idx} at centroid {centroid} and scaled by {dist}.")
    return x, centroid, dist


def apply_norm_unit_sphere(x, centroid, dist):
    return (x-centroid)/dist


def unnormalize_unit_sphere(x_norm, centroid, dist):
    return (x_norm * dist) + centroid


def scale_minmax(x, feature_idx=[3]):
    '''Min max scale select features within [0,1] range'''
    x_sel = x[..., feature_idx]
    xmin = np.min(x_sel, axis=tuple(np.arange(len(x.shape)-1)))
    xmax = np.max(x_sel, axis=tuple(np.arange(len(x.shape)-1)))
    x_sel = (x_sel-xmin)/(xmax-xmin)
    x[..., feature_idx] = x_sel
    logging.debug(f"Apply min max scaling on dim {feature_idx}.")
    return x


def random_sampling(lines, subsample=0.5, rng=0):
    '''Randomly sample a percentage of lines, return index'''
    if subsample > 1:
        nsample = min(subsample, len(lines))
    else:
        nsample = round(len(lines)*subsample)
    return random_select(lines, n_sample=nsample, rng=rng)


def farthest_sampling(lines, subsample=0.5, rng=0):
    '''Sample a percentage of lines using farthest 'point' sampling, return index'''
    ntotal = len(lines)
    if subsample > 1:
        nsample = min(subsample, len(lines))
    else:
        nsample = round(len(lines)*subsample)

    # shuffle before proceeding
    # (for array of the same length, shuffle the same way)
    lines = random_shuffle(lines, rng)
        
    # initalize parameters for state keeping
    remaining = np.arange(ntotal) # [P]
    sample_inds = np.zeros(nsample, dtype='int') # [S]
    dists = np.ones_like(remaining) # [P]
    
    # sample first point
    selected=0
    sample_inds[0] = remaining[selected]
    remaining = np.delete(remaining, selected) # [P - 1]
    
    for i in range(1, nsample):
        last_added = sample_inds[i-1]
        dist_to_last_added = np.array([mdf(lines[last_added], l) for l in lines[remaining]]) # [P - i]
        
        dists[remaining] = np.minimum(dist_to_last_added, dists[remaining])
        
        selected = np.argmax(dists[remaining])
        sample_inds[i] = remaining[selected]
        
        remaining = np.delete(remaining, selected)
    return sample_inds


def resample_lines_by_percent(lines, percent=0.5):
    '''Subsample each streamline with set percentage'''
    logging.debug(f"Resample each streamline to {100*percent}% of points.")
    ptct = [len(l) for l in lines]
    lines = [set_number_of_points(lines[i], int(percent*ptct[i])) for i in range(len(lines))]
    return ArraySequence(lines)


def get_bundle_idx(bname, bundle_idx):
    '''Retreiving indices of bundle'''
    if bname not in bundle_idx:
        logging.warning(f"Bundle {bname} does not exist in this subject.")
        return np.arange(0)
    indices = bundle_idx[bname]
    return np.arange(indices[0], indices[0]+indices[1])


def get_multibundle_idx(bundle_ls, bundle_idx):
    '''Retreiving indices of multiple bundles'''
    all_idx = np.arange(0)
    for bname in bundle_ls:
        idx = get_bundle_idx(bname, bundle_idx)
        all_idx = np.concatenate([all_idx, idx])
    return all_idx


def get_bundle_for_idx(line_idx, bundle_idx):
    '''Retrieve bundle name given a streamline index'''
    for bname, v in bundle_idx.items():
        if line_idx >= v[0] and line_idx < v[0]+v[1]:
            return bname


def recode_values(values, recode_type='label'):
    '''Numerically encode values with one-hot or label encoder'''
    if recode_type == 'onehot':
        enc = OneHotEncoder(handle_unknown='ignore')
        return enc.fit_transform(values.reshape(-1,1)).toarray(), enc
    else:
        enc = LabelEncoder()
        return enc.fit_transform(values), enc


def process_label(df, label_name, label_value, mode='enc'):
    '''Encode or decode label given a metadata dataframe'''
    if not isinstance(label_value, (list,pd.core.series.Series,np.ndarray)):
        label_value = np.array([label_value])

    if df[label_name].dtype != 'O': # can only label encode string value
        if label_name.lower() == 'age': # create age bins
            if mode == 'enc':
                value_min = df[label_name].min()
                return np.array((label_value-value_min) // 5).astype(np.int32)
            else:
                logging.warn('Does not support decoding age bins')
        return np.array(label_value)
    
    _, encoder = recode_values(df[label_name])
    if mode == 'enc':
        return encoder.transform(label_value).astype(np.int32)
    else:
        return encoder.inverse_transform(label_value)
    

def stratified_sampling(arr, stratify_by=None, n_sample=10, seed=0):
    '''Stratified sampling given values'''
    pct = n_sample if n_sample < 1 else n_sample/len(arr)        
    return train_test_split(range(len(arr)), test_size=pct, 
                            stratify=stratify_by,
                            random_state=seed)


def train_val_test_split(arr, stratify_by=None, train_pct=0.4, test_pct=0.3, seed=0):
    '''Stratified split into train/validation/test sets'''
    
    train_val, test = stratified_sampling(arr, stratify_by=stratify_by, 
                                          n_sample=test_pct, seed=seed)
    
    val_pct = 1-train_pct-test_pct
    stratify_by_val = stratify_by[train_val] if stratify_by is not None else None
    train, val = stratified_sampling(train_val, stratify_by=stratify_by_val, 
                                     n_sample=float(val_pct)/(1-test_pct), seed=seed)
    logging.info(f'Split array (N={len(arr)}) into train/val/test ({len(train)}/{len(val)}/{len(test)}) ({train_pct:.2f}/{val_pct:.2f}/{test_pct:.2f}).')
    return train, val, test
    

def get_df_column_value(df, key_val, key_col='Subject', value_col='Protocol', encode=False):
    '''
        Return column value corresponding to key value
        For example return protocol value for subject
    '''
    value = df.loc[df[key_col]==key_val, value_col].values[0]
    if encode:
        value = process_label(df=df, label_name=value_col, label_value=value)[0]
    return value