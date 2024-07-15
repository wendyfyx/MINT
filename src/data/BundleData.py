import logging
import numpy as np
from data.data_util import *

'''
    Class for loading on multiple bundles for each subject
    with support for subsampling, normalization and loading DTI
'''

class BundleData:
    
    def __init__(self, name, data_folder, subsample=None, dti=None, normalize=True, 
                 rec_bundles_folder="rec_bundles/", org_bundles_folder='org_bundles/', 
                 dti_folder="anatomical_measures/", **kwargs):
        
        self.name = name
        self.data_folder = data_folder
        self.normalize = normalize
        self.subsample = subsample

        self.feature_names = ['x','y','z']
        self.n_features = 3
        # self.data_args = kwargs
        self.kwargs=kwargs

        self.rec_bundles_folder = rec_bundles_folder
        self.org_bundles_folder = org_bundles_folder
        self.dti_folder = dti_folder
        self.X_encoded = None
        self.X_recon = None

        
        # Load streamlines
        if "parse_tract_func" in self.kwargs: # custom parsing function
            X, bundle_idx, ptct, subsample_idx = load_bundles(data_folder+name, 
                                                              subsample=self.subsample, 
                                                              sub_folder_path=self.rec_bundles_folder,
                                                              **self.kwargs)
        else: # default load from rec_bundles/ in MNI space (compatible with BUAN)
            X, bundle_idx, ptct, subsample_idx = load_bundles(data_folder+name, self.parse_tract_name, 
                                                              subsample=self.subsample, 
                                                              sub_folder_path=self.rec_bundles_folder,
                                                              **self.kwargs)
        self.X = X
        self.bundle_idx = bundle_idx # {bundle : [start, bundle_count]}
        if ptct is not None:
            self.ptct = np.array(ptct) # original number of points per line
        self.subsample_idx = subsample_idx # streamline indices selected in subsampling

        # Get bundle labels
        if self.X is None:
            self.y, self.bundle_num = None, None
            return
        y, bundle_num = make_y(self.bundle_idx) 
        self.y = y
        self.bundle_num = bundle_num # {index : bundle}
            
        # Load DTI
        if dti is not None and len(self.X)>0:
            self.n_features += len(dti)
            self.load_dtimap(dti, **self.kwargs)
        
        # Normalization by subject data
        if normalize and len(X)>0:
            # self.X_norm, self.centroid, self.dist = normalize_unit_sphere(self.X, **kwargs)
            self.X_norm, self.centroid, self.dist = self.normalize_unit_sphere(self.X)

        # del self.subsample_idx
        logging.info(f"Finished loading {self.name} with {len(self.bundle_idx)} bundles, \
                     {len(self.X)} data points and feautures {', '.join(self.feature_names)}.")


    def load_org_bundles(self):
        '''Load bundles in the native space for DTI mapping'''
        logging.info('Loading org_bundles for mapping DTI values.')

        self.kwargs.setdefault('rec_bundles_folder', self.org_bundles_folder)
        self.kwargs.setdefault('subsample', self.subsample_idx)
        self.kwargs.setdefault('dti', None)
        self.kwargs.setdefault('normalize', False) 
        return BundleData(self.name, self.data_folder, **self.kwargs)



    def load_dtimap(self, dti_measures, preprocess='3d', **kwargs):
        '''
            Load DTI mapped to tract, can load from multiple maps (FA, MD)
            with support for both 2D and 3D version of streamlines.
            Add result to X, and update feature names
        '''
        concat_axis = 2 if preprocess=='3d' else 1
        if self.org_bundles_folder is None:
            X_org = self.X
        else:
            bundle_org = self.load_org_bundles()
            X_org = bundle_org.X

        dtimap = []
        for dti in dti_measures:
            dti_fpath = glob.glob(f"{self.data_folder}{self.name}/{self.dti_folder}*{dti.lower()}*.nii.gz")[0]
            img, affine = load_nib(dti_fpath)
            logging.info(f'Loaded {dti.upper()} map from {dti_fpath} {img.shape}')

            bundle_mapped = dti2bundle(X_org, img, affine, is_2d=(preprocess=='2d'))
            dtimap.append(np.expand_dims(bundle_mapped, axis=concat_axis))
        dtimap = np.concatenate(dtimap, axis=concat_axis)

        self.X = np.concatenate((self.X, dtimap), axis=concat_axis)
        dti = [i.upper() for i in list(dti_measures)]
        self.feature_names.extend(dti)


    @staticmethod
    def normalize_unit_sphere(X):
        '''Normalize streamline coordinate to standard sphere'''
        centroid = [-2.8441544, -21.624807, 5.540867] # default to atlas brain centroid
        dist = 100 # preset radius
        return normalize_unit_sphere(X, feature_idx=[0,1,2], centroid=centroid, dist=dist)


    @staticmethod
    def scale_minmax(X):
        '''Normalization DTI, don't use on single subject'''
        return scale_minmax(X, feature_idx=np.arange(3, X.shape[-1]))


    def get_subj_bundle_idx(self, bname):
        '''
            Get indices of bundle in X
            Example usage: subj.X[subj.get_subj_bundle_idx('V')]
        '''
        return get_bundle_idx(bname, self.bundle_idx)


    def get_subj_multibundle_idx(self, bundle_ls):
        '''
            Get indices of multiple bundle in X
            Example usage: subj.X[subj.get_subj_multibundle_idx(['CST_L','CST_R'])]
        '''
        return get_multibundle_idx(bundle_ls, self.bundle_idx)


    def get_subj_bundle_for_idx(self, line_idx):
        '''
            Retrieve bundle name given a streamline index
            Example usage: subj.get_subj_bundle_for_idx(32)
        '''
        return get_bundle_for_idx(line_idx, self.bundle_idx)
    

    @staticmethod
    def parse_tract_name(fname):
        '''Parse tract name from filename, compatiable with BUAN output
           Custom function can be passed to __init__() for datasets with different naming formats.
        '''
        return fname.split('moved_')[1].split('__')[0]

            
    def print_summary(self):
        print(f"-----Data Summary-----")
        attrs = vars(self)
        for k, v in attrs.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {v}")
        print("-"*22)
