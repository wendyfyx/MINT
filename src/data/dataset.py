import logging
import h5py
import numpy as np
import pandas as pd
from itertools import chain
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset, Dataset, TensorDataset, DataLoader

from data.data_util import process_label, get_df_column_value
from data.fetcher import DataFetcher
from core.BundleName import name_to_int
from utils.file_util import load_json
from utils.general_util import random_select, random_select_consecutive, random_order, get_rng, seed_worker

class StreamlineDataset(IterableDataset):
    '''
        Iterable Dataset for loading streamlines from single .h5 file
        (Adapted from TractoLearn)
    '''

    def __init__(self, h5_data_fpath, batch_size=1, n_sample=1, n_step=10, 
                 feature_idx=None, consecutive=True, transform=None, 
                 subjects_include=None, bundles_include=None):
        super(StreamlineDataset).__init__()
        
        self.h5_data_fpath = h5_data_fpath
        self.subjects_include = subjects_include
        self.bundles_include = bundles_include

        # weighted sampling set up
        self.dict_count = load_json(f"{self.h5_data_fpath[:-3]}_count.json")
        self.dict_prob = self.calculate_prob()
        
        # Size of batch to generate per __next__() call
        self.batch_size = batch_size
        
        # To speed up data loading, multiple lines can be loaded 
        # from each bundle every time __next__() is called, note that
        # the batch would contain less diverse samples.
        self.n_sample = n_sample
        self.consecutive = consecutive

        # Define n_step batches as one epoch (mainly for progress bar)
        self.n_step = n_step
        
        self.feature_idx = feature_idx
        self.transform = transform
        self.rng = get_rng()

    def calculate_prob(self):
        d_sel = self.dict_count
        if self.subjects_include is not None:
            d_sel = {k:v for k,v in d_sel.items() if k.split('/')[0] in self.subjects_include}
        if self.bundles_include is not None:
            d_sel = {k:v for k,v in d_sel.items() if k.split('/')[1] in self.bundles_include}
        
        total = sum(d_sel.values())
        d_prob =  {k: v / total for k, v in d_sel.items()}
        return d_prob
        
    def set_seed(self, seed):
        self.rng = get_rng(seed)

    def _sample_from_bundle(self, data):
        if self.consecutive:
            idx = random_select_consecutive(data.shape[0], n_sample=self.n_sample, rng=self.rng)
        else:
            idx = np.sort(random_select(data, n_sample=self.n_sample, rng=self.rng))
        return idx

    def get_random_batch_weighted(self):
        keys = self.rng.choice(list(self.dict_prob.keys()), 
                               p=list(self.dict_prob.values()), 
                               size=self.batch_size)
        batch = []
        with h5py.File(self.h5_data_fpath, 'r') as hf:
            for k in keys:
                subj, bundle = k.split('/')
                data = hf[k]
                data = data[self._sample_from_bundle(data)]
                if self.feature_idx is not None:
                    data = data[..., self.feature_idx]
                if self.transform is not None:
                    data=self.transform(data)
                batch.append((data, name_to_int(bundle)))
        return batch
        
    def __len__(self):
        return self.n_step

    def __iter__(self):
        return self
        
    def __next__(self):
        return self.get_random_batch_weighted()


class StreamlineDatasetMulti(IterableDataset):
    '''
        Iterable Dataset for loading streamlines from .h5 data
        Support loading from multiple .h5 files using json config, and custom label given metadata
    '''

    def __init__(self, config_json, metadata_path=None, label_name=None, 
                 batch_size=10, n_sample=1, n_step=10, 
                 feature_idx=None, consecutive=True, transform=None):
        super().__init__()
        
        self.label_name = label_name
        self.df_meta = pd.read_csv(metadata_path) if metadata_path is not None else None

        # weighted sampling set up
        self.dict_config = load_json(config_json)
        self.dict_prob = self.calculate_prob()
        
        # Size of batch to generate per __next__() call
        self.batch_size = batch_size
        
        # To speed up data loading, multiple streamlines can be loaded 
        # from each bundle every time __next__() is called, note that
        # the batch would contain less diverse samples.
        self.n_sample = n_sample
        self.consecutive = consecutive

        # Define n_step batches as one epoch (mainly for progress bar)
        self.n_step = n_step
        
        self.feature_idx = feature_idx
        self.transform = transform
        self.rng = get_rng()

    def calculate_prob(self):
        d_sel = {k:v['Count'] for k, v in self.dict_config.items()}        
        total = sum(d_sel.values())
        d_prob =  {k: v / total for k, v in d_sel.items()}
        return d_prob
        
    def set_seed(self, seed):
        self.rng = get_rng(seed)

    def _sample_from_bundle(self, data):
        if self.consecutive:
            idx = random_select_consecutive(data.shape[0], n_sample=self.n_sample, rng=self.rng)
        else:
            idx = np.sort(random_select(data, n_sample=self.n_sample, rng=self.rng))
        return idx

    def _get_keys(self):
        keys = self.rng.choice(list(self.dict_prob.keys()), 
                               p=list(self.dict_prob.values()), 
                               size=self.batch_size)
        d={}
        for i in keys:
            ss = i.rsplit('/', 2)
            k = ss[0] # h5 path
            v = '/'.join(ss[1:]) # subj/bundle
            d.setdefault(k, []).append(v)
        return d

    def get_random_batch_weighted(self):
        keys = self._get_keys()
        batch = []
        for fpath, k in keys.items(): # iterate over datasets in the config
            with h5py.File(fpath, 'r') as hf:
                for k1 in k: # iterate over bundles in the dataset
                    subj, bundle = k1.split('/')

                    # Set label
                    if self.label_name is None or self.df_meta is None:
                        label = np.array([name_to_int(bundle)])
                    else:
                        label = []
                        for lname in self.label_name:
                            label_value = get_df_column_value(self.df_meta, subj, key_col='Subject', value_col=lname)
                            label.append(process_label(self.df_meta, lname, label_value, mode='enc'))
                        label = np.concatenate(label, axis=0)
                    # Sample from bundle
                    data = hf[f"{subj}/{bundle}"]
                    data = data[self._sample_from_bundle(data)]

                    # Apply feature indexing and transform
                    if self.feature_idx is not None:
                        data = data[..., self.feature_idx]
                    if self.transform is not None:
                        data=self.transform(data)

                    batch.append((data, label))
        return batch
        
    def __len__(self):
        return self.n_step

    def __iter__(self):
        return self
        
    def __next__(self):
        return self.get_random_batch_weighted()
    

class StreamlineCollator:
    '''
        Custom collator for StreamlineDataset
        Returns the streamline data and bundle labels (integer) defined by Atlas30
    '''
    def __init__(self, n_sample=1, seed=0):
        self.n_sample = n_sample
        self.rng=get_rng(seed)

    def __call__(self, batch):
        batch = list(chain(*batch))
        batch_data, batch_labels = torch.utils.data.dataloader.default_collate(batch)
        # Reshape input data (preserve last two dimension)
        dims = batch_data.size()[2:]
        batch_data = batch_data.view(-1, *dims)
        # Convert label to int
        batch_labels = batch_labels.repeat_interleave(self.n_sample, dim=0)
        # Shuffle batch
        new_idx = random_order(len(batch_data), rng=self.rng)
        return batch_data[new_idx], batch_labels[new_idx]
    

class StreamlineMapDataset(Dataset):
    '''
        Map-style dataset class for numpy array loaded in memory
    '''
    def __init__(self, h5_data_fpath, feature_idx=None, transform=None, apply_bootstrap=False,
                 subjects_include=None, bundles_include=None, seed=0):
        self.fetcher = DataFetcher(h5_data_fpath)
        self.feature_idx = feature_idx
        self.transform = transform
        self.apply_bootstrap = apply_bootstrap
        
        self.subjects_include = self.fetcher.fetch_subject_names() if subjects_include is None else subjects_include
        self.bundles_include = bundles_include
        self.rng = get_rng(seed)
        self.get_data()

    def get_data(self):
        if self.apply_bootstrap:
            self.subjs_bootstrap = self.rng.choice(self.subjects_include, 
                                                replace=True, 
                                                size=len(self.subjects_include))
            logging.info(f"Creating dataset with bootstrap from {len(self.subjs_bootstrap)} subjects. ({self.subjs_bootstrap})")
        else:
            self.subjs_bootstrap = self.subjects_include
            logging.info(f"Creating dataset without boostrap from {len(self.subjs_bootstrap)} subjects. ({self.subjs_bootstrap})")
        
        X = []
        y = []
        for s in self.subjs_bootstrap:
            Xs, ys = self.fetcher.fetch_bundles_for_subj(s, bundles=self.bundles_include, load_y=True)
            if len(Xs)==0:
                continue
            if self.feature_idx is not None:
                Xs = Xs[...,self.feature_idx]
            X.append(Xs)
            y.append(ys)
        self.X = np.concatenate(X, axis=0)
        self.y = np.concatenate(y, axis=0)
        logging.info(f"Final dataset X {self.X.shape} and y {self.y.shape}.")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x=self.transform(x)
        return x.squeeze(), y


class StreamlineMapDatasetMulti(Dataset):
    '''
        Map-style dataset class for numpy array loaded in memory
        Support loading from multiple .h5 files using json config
    '''
    def __init__(self, config_json, metadata_path=None, label_name=None,
                 feature_idx=None, transform=None, apply_bootstrap=False,
                 seed=0):
        
        self.label_name = label_name
        self.df_meta = pd.read_csv(metadata_path) if metadata_path is not None else None
        self.dict_config = load_json(config_json)
        
        self.feature_idx = feature_idx
        self.transform = transform
        self.apply_bootstrap = apply_bootstrap
        
        # self.subjects_include = self._get_subjs()
        # self.bundles_include = bundles_include
        self.rng = get_rng(seed)
        self.get_data()
    
    def _get_subjs(self):
        return list(set([i.rsplit('/',2)[1] for i in self.dict_config.keys()]))

    def get_data(self):
        X = []
        y = []
        for fpath, k in process_json_config(self.dict_config).items():
            fetcher = DataFetcher(fpath)
            for subj, bundles in k.items():
                Xs, ys = fetcher.fetch_bundles_for_subj(subj, bundles=bundles, load_y=True)
                if self.label_name is not None and self.df_meta is not None:
                    label = []
                    for lname in self.label_name:
                        label_value = get_df_column_value(self.df_meta, subj, key_col='Subject', value_col=lname)
                        label.append(process_label(self.df_meta, lname, label_value, mode='enc')[0])
                    ys = np.tile(np.array(label), [len(Xs),1])
                if len(Xs)==0:
                    continue
                if self.feature_idx is not None:
                    Xs = Xs[...,self.feature_idx]
                X.append(Xs)
                y.append(ys)
        self.X = np.concatenate(X, axis=0)
        self.y = np.squeeze(np.concatenate(y, axis=0))
        logging.info(f"Final dataset X {self.X.shape} and y {self.y.shape}.")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x=self.transform(x)
        return x.squeeze(), y


def make_data_loader(data, *args, seed=0, batch_size=512, 
                          shuffle=True, num_workers=0):
    '''Make data loader for map-stype tensor dataset'''
    g_seed = torch.Generator()
    g_seed.manual_seed(seed)
    dataset = TensorDataset(data, *args)
    return DataLoader(dataset, batch_size=batch_size,
                    shuffle=shuffle, num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    generator=g_seed)


def process_json_config(d_config):
    d = defaultdict(lambda: defaultdict(list))
    for i in d_config.keys():
        ss = i.rsplit('/', 2)
        fpath = ss[0]
        subj, bundle = ss[1:]
        d[fpath][subj].append(bundle)
    return d