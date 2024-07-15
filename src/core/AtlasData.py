import logging
from warnings import warn

from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk
# from dipy.io.surface import load_gifti
from utils.file_util import load_pickle

# DIPY_ATLAS="~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/"
DIPY_ATLAS="/ifs/loni/faculty/thompson/four_d/wfeng/Datasets/Atlas/230615-Atlas-38-Bundles/bundles"
MNI_ATLAS="../../assets/"

class AtlasData:
    def __init__(self, bundle_atlas_folder=DIPY_ATLAS, mni_atlas_folder=MNI_ATLAS):
        self.bundle_atlas_folder = bundle_atlas_folder
        self.mni_atlas_folder = mni_atlas_folder
    
    def fetch_bundle(self, bundle):
        fpath = f"{self.bundle_atlas_folder}/{bundle}.trk"
        return load_trk(fpath, "same", bbox_valid_check=False).streamlines
    
    def fetch_cam_setting(self, bundle):
        fpath = f"{self.mni_atlas_folder}/cam_settings/cam_{bundle}_glass_brain.pkl"
        return load_pickle(fpath)
    
    def fetch_glass_brain(self):
        fpath = f"{self.mni_atlas_folder}/glass_brain.nii.gz"
        data, affine = load_nifti(fpath)
        logging.info(f"Fetched glass brain from {fpath}.")
        return data, affine
    
    def fetch_pial_surf_l(self):
        fpath = f"{self.mni_atlas_folder}/pial_left.gii.gz"
        surface = load_gifti(fpath)
        vertices, faces = surface
        logging.info(f"Fetched left pial surface from {fpath}.")
        return vertices, faces

    def fetch_pial_surf_r(self):
        fpath = f"{self.mni_atlas_folder}/pial_right.gii.gz"
        surface = load_gifti(fpath)
        vertices, faces = surface
        logging.info(f"Fetched right pial surface from {fpath}.")
        return vertices, faces