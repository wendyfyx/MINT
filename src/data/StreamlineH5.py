import logging
import numpy as np
import h5py

from data.BundleData import BundleData
from core.BundleName import Atlas30, name_to_int

class StreamlineH5:
    def __init__(self, h5path, subjs, **kwargs):
        self.h5path = h5path
        self.subjs = subjs
        self.kwargs = kwargs

    def create_h5_files(self, mode='a', n_lines_per_chunk=32):
        '''Create .h5 files for bundle data'''
        with h5py.File(self.h5path, mode) as hf:
            logging.info(f"Creating H5 dataset at {self.h5path} (mode={mode})...")
            for sub in self.subjs:
                grp = hf.create_group(sub) if sub not in hf else hf[sub]
                subjdata = BundleData(sub, **self.kwargs)
                chunk_size = (n_lines_per_chunk, self.kwargs.get('n_points'), subjdata.n_features)
                for bundle in subjdata.bundle_idx.keys():
                    if f"{sub}/{bundle}" not in hf:
                        x = subjdata.X[subjdata.get_subj_bundle_idx(bundle)].astype('float32')
                        logging.info(f"Creating {sub}/{bundle} of shape {x.shape} and chunk size of {chunk_size}")
                        if len(x)<1:
                            grp.create_dataset(bundle, data=h5py.Empty("f"))
                        else:
                            # Chunking is enabled for more efficient read
                            grp.create_dataset(bundle, data=x, compression="gzip", chunks=chunk_size)
                    else:
                        logging.info(f"{sub}/{bundle} already exists...")

    def _get_BundleData_test(self, sub):
        return BundleData(sub, **self.kwargs)

    def get_bundle_data(self, subject, bundle):
        '''Get bundle from subject'''
        k = f"{subject}/{bundle}"
        with h5py.File(self.h5path, 'r') as hf: 
            if k not in hf:
                return None
            dst = hf[k]
            if dst.shape is None:
                return None
            return dst[...]
        
    def get_subject_names(self):
        '''Get names of all subjects'''
        with h5py.File(self.h5path, 'r') as hf:
            return list(hf.keys())
    
    def get_bundle_names_for_subject(self, subject):
        '''Get all bundles available given subject name'''
        with h5py.File(self.h5path, 'r') as hf:
            if subject not in hf:
                logging.warning(f"{subject} does not exist in file")
                return
            return list(hf[subject].keys())
        
    def get_bundle_data_for_subj(self, subject, bundles=None, load_y=False):
        X = []
        y = []
        with h5py.File(self.h5path, 'r') as hf:
            bundles = hf[subject].keys() if bundles is None else [b for b in bundles if b in hf[subject].keys()]
            logging.debug(f"Fetching {len(bundles)} bundles for subject {subject} ({bundles}).")
            for k in bundles:
                if k not in Atlas30.__members__:
                    logging.warn(f"[{k}] is not a defined bundle.")
                    continue
                data = hf[f'{subject}/{k}'][...]
                if load_y:
                    y.extend([name_to_int(k)]*len(data))
                X.append(data)
                logging.debug(f"Fetched {k} for subject {subject} {data.shape}.")
            if len(X)>0:
                X = np.concatenate(X, axis=0)
                logging.info(f"Fetched {len(bundles)} bundles {X.shape} for subject {subject}.")
            if load_y:
                y = np.array(y)
                return X, y
            return X, None