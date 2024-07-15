import numpy as np
from utils.general_util import get_rng


class RandomFlip(object):
    '''Randomly flip streamlines'''
    def __init__(self, prob=0.5, rng=0):
        self.prob = prob
        self.rng = get_rng(rng)

    @staticmethod
    def flip_streamline(line, rng, prob=0.5):
        if rng.uniform() > prob:
            return np.flip(line, axis=0)
        else:
            return line

    def random_flip(self, lines):
        lines = np.array([self.flip_streamline(line, self.rng, self.prob) for line in lines])
        return lines

    def __call__(self, sample):
        if len(sample.shape)<3:
            sample=np.expand_dims(sample, 0)
        return self.random_flip(sample)


class Normalize:
    '''Normalize to unit sphere given centroid and radius'''
    def __init__(self, centroid='atlas', radius='atlas', feature_idx=[0,1,2,3,4,5,6]):
        centroid_atlas = [-2.8441544, -21.624807, 5.540867] # default to atlas brain centroid
        rad_atlas = 100 # preset radius
        self.centroid = centroid_atlas if centroid=='atlas' else centroid
        self.radius = rad_atlas if radius=='atlas' else radius
        self.feature_idx = [0,1,2,3,4,5,6] if feature_idx is None else feature_idx
        self.dti_tmin = -1
        self.dti_tmax = 1

    @staticmethod
    def normalize_unit_sphere(x, feature_idx=[0,1,2], centroid=None, radius=None):
        if feature_idx is None:
            return x
        x = x.copy()
        x_sel = x[..., feature_idx]
        if centroid is not None:
            x_sel -= centroid
        if radius is not None:
            x_sel /= radius
        x[..., feature_idx] = x_sel
        return x

    @staticmethod
    def unnormalize_unit_sphere(x, feature_idx=[0,1,2], centroid=None, radius=None):
        if feature_idx is None:
            return x
        x = x.copy()
        x_sel = x[..., feature_idx]
        if radius is not None:
            x_sel *= radius
        if centroid is not None:
            x_sel += centroid
        x[..., feature_idx] = x_sel
        return x
    
    @staticmethod
    def scale_to_range(x, feature_idx=[4,5,6], rmin=0, rmax=0.01, vmin=0, vmax=1):
        if feature_idx is None:
            return x
        x = x.copy()
        x_sel = x[..., feature_idx]
        x_sel = (vmax-vmin) * ((x_sel-rmin)/(rmax-rmin)) + vmin
        x[..., feature_idx] = x_sel
        return x
    
    @staticmethod
    def unscale(x, feature_idx=[4,5,6], rmin=0, rmax=0.01, vmin=0, vmax=1):
        if feature_idx is None:
            return x
        x = x.copy()
        x_sel = x[..., feature_idx]
        x_sel = (rmax-rmin) *((x_sel-vmin)/(vmax-vmin)) + rmin
        x[..., feature_idx] = x_sel
        return x
    
    def _get_fa_idx(self):
        return self.feature_idx.index(3) if 3 in self.feature_idx else None

    def _get_md_idx(self):
        return self.feature_idx.index(4) if 4 in self.feature_idx else None
    
    def _get_rd_idx(self):
        return self.feature_idx.index(5) if 5 in self.feature_idx else None
    
    def _get_ad_idx(self):
        return self.feature_idx.index(6) if 6 in self.feature_idx else None
    
    def __call__(self, sample):
        # X, Y, Z
        sample = self.normalize_unit_sphere(sample, feature_idx=[0,1,2], 
                                          centroid=self.centroid, radius=self.radius)
        # FA
        sample = self.scale_to_range(sample, feature_idx=self._get_fa_idx(), 
                                     rmin=0, rmax=1, 
                                     vmin=self.dti_tmin, vmax=self.dti_tmax)
        # MD
        sample = self.scale_to_range(sample, feature_idx=self._get_md_idx(), 
                                     rmin=0, rmax=0.001, 
                                     vmin=self.dti_tmin, vmax=self.dti_tmax)
        # RD
        sample = self.scale_to_range(sample, feature_idx=self._get_rd_idx(), 
                                     rmin=0, rmax=0.001, 
                                     vmin=self.dti_tmin, vmax=self.dti_tmax)
        # AD
        sample = self.scale_to_range(sample, feature_idx=self._get_ad_idx(), 
                                     rmin=0, rmax=0.003, 
                                     vmin=self.dti_tmin, vmax=self.dti_tmax)

        return sample

    def unnormalize(self, sample):
        # X, Y, Z
        sample = self.unnormalize_unit_sphere(sample, feature_idx=[0,1,2], 
                                            centroid=self.centroid, radius=self.radius)
        # FA
        sample = self.unscale(sample, feature_idx=self._get_fa_idx(), 
                              rmin=0, rmax=1, 
                              vmin=self.dti_tmin, vmax=self.dti_tmax)
        # MD
        sample = self.unscale(sample, feature_idx=self._get_md_idx(), 
                              rmin=0, rmax=0.001, 
                              vmin=self.dti_tmin, vmax=self.dti_tmax)
        # RD
        sample = self.unscale(sample, feature_idx=self._get_rd_idx(), 
                              rmin=0, rmax=0.001, 
                              vmin=self.dti_tmin, vmax=self.dti_tmax)
        # AD
        sample = self.unscale(sample, feature_idx=self._get_ad_idx(), 
                              rmin=0, rmax=0.003, 
                              vmin=self.dti_tmin, vmax=self.dti_tmax)
        return sample