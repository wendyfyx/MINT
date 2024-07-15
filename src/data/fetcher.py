import logging

import glob
from enum import Enum

import numpy as np

from core.BundleName import name_to_int
from data.BundleData import BundleData
from data.StreamlineH5 import StreamlineH5


class DataFetcher:
    def __init__(self, h5_path, data_path=None, split=None, subjects=None, bundles=None, **kwargs):
        self.h5_path=h5_path
        self.data_path = data_path
        self.subjects = subjects

        if self.data_path is not None:
            if not self.data_path.endswith("/"):
                self.data_path = self.data_path + "/"
            if split is not None:
                self.data_folder = f"{self.data_path}{split}/"
            else:
                self.data_folder = self.data_path
            logging.info(f"FETCHING FROM {self.data_folder}")

        if self.subjects is not None:
            self.subjects=sorted([i.rsplit('/', 1)[1] for i in glob.glob(f"{self.data_folder}*")]) if self.subjects=='all' else self.subjects
            logging.info(f"TOTAL {len(self.subjects)} SUBJECTS: {self.subjects}")

        self.bundles=bundles
        self.kwargs=kwargs
        self.set_args()

        self.h5data=StreamlineH5(h5path=self.h5_path, subjs=self.subjects, **self.kwargs)

    def set_args(self):
        pass

    @staticmethod
    def parse_tract_name(fname):
        return fname.split('moved_')[1].split('__')[0]

    def create_dataset(self, **kwargs):
        if self.subjects is None:
            logging.warn("Empty subjects, cannot create dataset!")
            return
        self.h5data.create_h5_files(**kwargs)

    def fetch(self, subj, bundle):
        '''Fetch bundle data for subject.'''
        return self.h5data.get_bundle_data(subj, bundle)
    
    def fetch_subject_names(self):
        '''Fetch all available subjects' names.'''
        return self.h5data.get_subject_names()
    
    def fetch_bundles_names(self, subj):
        '''Fetch all available bundle names for subject.'''
        return self.h5data.get_bundle_names_for_subject(subj)
    
    def fetch_bundles_for_subj(self, subj, **kwargs):
        '''Fetch all bundles' content for subject, returns both streamlines and labels.'''
        return self.h5data.get_bundle_data_for_subj(subj, **kwargs)

    def fetch_bundledata(self, subj):
        '''Fetch bundles for subject using BundleData class using specified kwargs.'''
        return BundleData(subj, **self.kwargs)
    
    
class TractoInfernoFetcher(DataFetcher):
    def __init__(self, h5_data_path, split='trainset', subjects=None, bundles=None, **kwargs):
        data_path="/ifs/loni/faculty/thompson/four_d/wfeng/Datasets/221208-TractInferno/derivatives"
        super(TractoInfernoFetcher, self).__init__(h5_data_path, data_path, 
                                                   split=split, subjects=subjects, bundles=bundles, 
                                                   **kwargs)

    def set_args(self):
        self.kwargs.setdefault('rec_bundles_folder', 'recobundles/rec_bundles/')
        self.kwargs.setdefault('org_bundles_folder', 'recobundles/org_bundles/')
        self.kwargs.setdefault('dti_folder', 'dti/')
        self.kwargs.setdefault('parse_tract_func', self.parse_tract_name)
        self.kwargs.setdefault('data_folder', self.data_folder)
        if self.bundles is not None:
            self.kwargs.setdefault('tracts_include', self.bundles)
    

class MultisitePDFetcher(DataFetcher):
    def __init__(self, h5_data_path, split='control', subjects=None, bundles=None, **kwargs):
        data_path="/ifs/loni/faculty/thompson/four_d/cowenswalton/TRACTOGRAMS/FRONTIERS/data"
        super(MultisitePDFetcher, self).__init__(h5_data_path, data_path, 
                                                 split=split, subjects=subjects, bundles=bundles, 
                                                 **kwargs)

    def set_args(self):
        self.kwargs.setdefault('rec_bundles_folder', 'rec_bundles/')
        self.kwargs.setdefault('org_bundles_folder', 'org_bundles/')
        self.kwargs.setdefault('dti_folder', 'anatomical_measures/')
        self.kwargs.setdefault('parse_tract_func', self.parse_tract_name)
        self.kwargs.setdefault('data_folder', self.data_folder)
        self.kwargs.setdefault('tracts_exclude', ['CST_L_s', 'CST_R_s'])
        if self.bundles is not None:
            self.kwargs.setdefault('tracts_include', self.bundles)
