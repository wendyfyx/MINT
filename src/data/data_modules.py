import logging 
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchvision import transforms

from data.dataset import StreamlineDataset, StreamlineDatasetMulti, StreamlineCollator, StreamlineMapDataset, StreamlineMapDatasetMulti
from data.transform import RandomFlip, Normalize
from utils.general_util import get_rng, set_seed_pl, seed_worker


class StreamlineDataModuleBase(pl.LightningDataModule):
    '''Base class of loading streamline data built with pytorch-lightning data module'''
    def __init__(self, h5_data_path=None, config_path=None, 
                 metadata_path=None, label_name=None,
                 batch_size=64, n_worker=4,
                 feature_idx=None, apply_centering=True, apply_scaling=True, 
                 subjects_include=None, bundles_include=None, seed=0):
        super().__init__()

        # Parameters for dataset with single h5 file
        self.h5_data_path = h5_data_path
        self.subjects_include = subjects_include
        self.bundles_include = bundles_include

        # Parameters for dataset with json configs
        self.config_path = config_path
        self.metadata_path = metadata_path
        self.label_name = label_name
        
        # Parameters for data transform
        self.feature_idx = feature_idx
        self.apply_centering = apply_centering
        self.apply_scaling = apply_scaling

        # General data loading parameters
        self.batch_size = batch_size
        self.n_worker = n_worker
        self.seed = seed            

        # Lightning data module set up
        self.prepare_data_per_node = True
        self.save_hyperparameters()

    def get_transforms(self):
        '''Default transform: random flipping and unit sphere scaling using atlas data'''
        centroid = 'atlas' if self.apply_centering else None
        radius = 'atlas' if self.apply_scaling else None
        return transforms.Compose([RandomFlip(prob=0.5, rng=get_rng(self.seed)),
                                   Normalize(centroid=centroid, radius=radius, 
                                             feature_idx=self.feature_idx)])
    
    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_dataset = self.make_dataset()
    
    def train_dataloader(self):
        return self.make_dataloader(self.train_dataset)    


class StreamlineDataModule(StreamlineDataModuleBase):
    '''
        Lightning data module for loading streamline data from single .h5 file 
        using iterable dataset
    '''
    def __init__(self, h5_data_path, batch_size=64,
                 n_sample=1, n_loader=100, 
                 n_worker=4, feature_idx=None, 
                 apply_centering=True, apply_scaling=True, 
                 subjects_include=None, bundles_include=None,
                 seed=0):
        
        self.n_sample = n_sample
        self.n_step = n_loader * n_sample #* batch_size
    
        super().__init__(h5_data_path, batch_size=batch_size, n_worker=n_worker, 
                         feature_idx=feature_idx, apply_centering=apply_centering, apply_scaling=apply_scaling, 
                         subjects_include=subjects_include, bundles_include=bundles_include, seed=seed)


    @staticmethod
    def worker_init_fn(worker_id):
        '''Initialize worker with worker_id as seed'''
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.set_seed(worker_id)
        
    def make_dataset(self):
        '''Make dataset with random flipping and normalization'''
        return StreamlineDataset(self.h5_data_path, batch_size=self.batch_size,
                                 n_sample=self.n_sample, n_step=self.n_step, 
                                 feature_idx=self.feature_idx,
                                 consecutive=True, transform=self.get_transforms(), 
                                 subjects_include=self.subjects_include,
                                 bundles_include=self.bundles_include)
    
    def make_dataloader(self, dataset):
        '''Make dataloader given dataset'''
        persistent = True if self.n_worker > 0 else False
        return DataLoader(dataset, batch_size=1,
                          collate_fn=StreamlineCollator(n_sample=self.n_sample),
                          num_workers=self.n_worker,
                          worker_init_fn=self.worker_init_fn, 
                          persistent_workers=persistent, pin_memory=True)
    

class StreamlineDataModuleMulti(StreamlineDataModuleBase):
    def __init__(self, config_path, metadata_path=None, label_name=None,
                 batch_size=64, n_sample=1, n_loader=100, 
                 n_worker=4, feature_idx=None, 
                 apply_centering=True, apply_scaling=True, 
                 seed=0):
        
        self.n_sample = n_sample
        self.n_step = n_loader * n_sample #* batch_size
    
        super().__init__(config_path=config_path, metadata_path=metadata_path, label_name=label_name,
                         batch_size=batch_size, n_worker=n_worker, feature_idx=feature_idx, 
                         apply_centering=apply_centering, apply_scaling=apply_scaling,
                         seed=seed)

    @staticmethod
    def worker_init_fn(worker_id):
        '''Initialize worker with worker_id as seed'''
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.set_seed(worker_id)


    def make_dataset(self):
        '''Make dataset with random flipping and normalization'''
        return StreamlineDatasetMulti(self.config_path, self.metadata_path, self.label_name,
                                      batch_size=self.batch_size,
                                      n_sample=self.n_sample, n_step=self.n_step, 
                                      feature_idx=self.feature_idx, consecutive=True, 
                                      transform=self.get_transforms())
    
    def make_dataloader(self, dataset):
        '''Make dataloader given dataset'''
        persistent = True if self.n_worker > 0 else False
        return DataLoader(dataset, batch_size=1,
                          collate_fn=StreamlineCollator(n_sample=self.n_sample),
                          num_workers=self.n_worker,
                          worker_init_fn=self.worker_init_fn, 
                          persistent_workers=persistent, pin_memory=True)


class StreamlineDataModuleMap(StreamlineDataModuleBase):
    '''
        Lightning data module for loading streamline data from single .h5 file 
        using map-style dataset with support for bootstrapping
    '''
    def __init__(self, h5_data_path, batch_size=64,
                 n_worker=4, feature_idx=None, apply_bootstrap=False,
                 apply_centering=True, apply_scaling=True, 
                 subjects_include=None, bundles_include=None, seed=0):
        
        self.apply_bootstrap = apply_bootstrap

        super().__init__(h5_data_path, batch_size=batch_size, n_worker=n_worker, 
                         feature_idx=feature_idx, apply_centering=apply_centering, apply_scaling=apply_scaling, 
                         subjects_include=subjects_include, bundles_include=bundles_include, seed=seed)

    def make_dataset(self):
        return StreamlineMapDataset(self.h5_data_path, 
                                    feature_idx=self.feature_idx, 
                                    transform=self.get_transforms(),
                                    apply_bootstrap=self.apply_bootstrap,
                                    subjects_include=self.subjects_include,
                                    bundles_include=self.bundles_include, 
                                    seed=self.seed)

    def make_dataloader(self, dataset):
        '''Make dataloader given dataset'''
        persistent = True if self.n_worker > 0 else False
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_worker,
                          worker_init_fn=seed_worker, 
                          persistent_workers=persistent)
    

class StreamlineDataModuleMapMulti(StreamlineDataModuleBase):
    '''
        Lightning data module for loading streamline data from single .h5 file 
        using map-style dataset with support for bootstrapping
    '''
    def __init__(self, config_path, metadata_path=None, label_name=None, 
                 batch_size=64, n_worker=4, feature_idx=None, apply_bootstrap=False,
                 apply_centering=True, apply_scaling=True, seed=0):
        
        self.apply_bootstrap = apply_bootstrap

        super().__init__(config_path=config_path, metadata_path=metadata_path, label_name=label_name,
                         batch_size=batch_size, n_worker=n_worker, feature_idx=feature_idx, 
                         apply_centering=apply_centering, apply_scaling=apply_scaling, 
                         seed=seed)

    def make_dataset(self):
        return StreamlineMapDatasetMulti(self.config_path, self.metadata_path, self.label_name,
                                         feature_idx=self.feature_idx, 
                                         transform=self.get_transforms(),
                                         apply_bootstrap=self.apply_bootstrap,
                                         seed=self.seed)

    def make_dataloader(self, dataset):
        '''Make dataloader given dataset'''
        persistent = True if self.n_worker > 0 else False
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_worker,
                          worker_init_fn=seed_worker, 
                          persistent_workers=persistent)