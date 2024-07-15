import logging
from functools import cached_property

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
from PIL import Image

from dipy.viz import window, actor
from dipy.stats.analysis import assignment_map
from nibabel.streamlines.array_sequence import ArraySequence

from core.AtlasData import AtlasData
from utils.plot_util import autocrop

def get_segment_idx(bundle, ref_bundle, n_segments=100):
    if isinstance(bundle, np.ndarray):
        bundle = ArraySequence(bundle[:,:,:3])
    indx = assignment_map(bundle, ref_bundle, n_segments)
    indx = np.array(indx)
    return indx


def value2color(values, 
                vmin=None, 
                vmax=None, 
                vcenter=None,
                cmap_name='viridis',
                label='values', 
                plot_cmap=False, 
                save_cmap=None, 
                **kwargs):
    
    '''Map values to color using colormap'''
    
    # Get value range
    if vmin is None:
        vmin = np.percentile(values, 5) #min(values)
    if vmax is None:
        vmax = np.percentile(values, 95) #max(values)
        
    # Map to colormap
    if vcenter is not None and vmin < vcenter:
        maxdist = max(abs(vmin - vcenter), abs(vmax - vcenter))
        vmin = vcenter - maxdist
        vmax = vcenter + maxdist
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    logging.info(f'vmin = {vmin}, vcenter = {vcenter}, vmax={vmax}.')
    cmap = plt.get_cmap(cmap_name)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = cmap.to_rgba(values)[:,:3]
    
    # Plot and/or save colormap
    if plot_cmap:
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        cb = fig.colorbar(cmap, cax=ax, orientation='horizontal')
        cb.set_label(label=label, size=14)
        if save_cmap is not None:
            fig.savefig(save_cmap)
    return colors


def value2diskcolor(bundle, values, ref_bundle=None, indx=None, **kwargs):
    '''Map along-tract values to color using colormap'''
    colors = value2color(values, **kwargs)
    colors = [tuple(i) for i in list(colors)]
    ref_bundle = bundle if ref_bundle is None else ref_bundle
    indx = get_segment_idx(bundle, ref_bundle, n_segments=len(values)) if indx is None else indx
    disks_color = []
    for i in range(len(indx)):
        disks_color.append(tuple(colors[indx[i]]))
    return disks_color
    
    
class BundleVisualizer:
    def __init__(self, data, feature_idx=[0,1,2,3,4,5,6],
                  bundle_type=None, is_local=True):
        self.data = data
        self.bundle_type = bundle_type
        features_names = ['x','y','z','fa','md','rd','ad','segment']
        self.features = [features_names[i] for i in feature_idx]
        
        if isinstance(data, np.ndarray):
            self.bundle = self.data[:,:,:3]
        else:
            self.bundle = self.data
        self.atlas = AtlasData(is_local=is_local)
        
    def get_metric(self, metric):
        '''Get metric value given feature names'''
        idx = self.features.index(metric)
        if isinstance(self.data, np.ndarray):
            return self.data[:,:,idx]
        else:
            return None
    
    @cached_property
    def glass_brain_actor(self):
        '''Get ROI actor for glass brain'''
        gb, affine = self.atlas.fetch_glass_brain()
        gb_actor = actor.contour_from_roi(gb, affine=affine,
                                       color=np.array([0, 0, 0]),
                                       opacity=0.06)
        return gb_actor

    @cached_property
    def pial_surf_l_actor(self):
        '''Get surface actor for left pial surface'''
        vertices, faces = self.atlas.fetch_pial_surf_l()
        colors = np.zeros((vertices.shape[0], 3))
        surf_actor = actor.surface(vertices, faces=faces, colors=colors)
        surf_actor.GetProperty().SetOpacity(0.06)
        return surf_actor
    
    @cached_property
    def pial_surf_r_actor(self):
        '''Get surface actor for right pial surface'''
        vertices, faces = self.atlas.fetch_pial_surf_r()
        colors = np.zeros((vertices.shape[0], 3))
        surf_actor = actor.surface(vertices, faces=faces, colors=colors)
        surf_actor.GetProperty().SetOpacity(0.06)
        return surf_actor

    @cached_property
    def bundle_atlas(self):
        '''Get atlas bundle'''
        return self.atlas.fetch_bundle(self.bundle_type)
    
    @cached_property
    def assignment_map(self, n_segments=100):
        '''Get index for along-tract segments'''
        return get_segment_idx(self.bundle, self.bundle_atlas, n_segments=n_segments)
    

    def get_stream_actor(self, bundle, colors=None, plot_points=False):
        if plot_points:
            stream_actor = actor.point(bundle.reshape(-1, 3), 
                                       colors=window.colors.coral, point_radius=3e-3)
        else:
            stream_actor = actor.line(self.bundle, 
                                      fake_tube=True, linewidth=6, colors=colors)
        return stream_actor
    
        
    def visualize_bundle(self, colors=None, 
                         plot_glass_brain=False,
                         plot_pial_surf=False,
                         interactive=False, # show interactive window
                         plot_points=False, # plot lines as points
                         save_to=None, # save image
                         do_plot=True, # plot output
                         ret_img=False, # return image as an array
                         apply_autocrop=True, # automatically crop image
                         **kwargs):
        
        # Set up
        scene = window.Scene()
        scene.SetBackground(1, 1, 1)
        if self.bundle_type is not None:
            self.cam_settings = self.atlas.fetch_cam_setting(self.bundle_type, 
                                                             glass_brain=(plot_pial_surf or plot_glass_brain))
            scene.set_camera(position=self.cam_settings['pos'], 
                            focal_point=self.cam_settings['foc'],
                            view_up=self.cam_settings['vup'])

        # Glass brain
        if plot_glass_brain:
            scene.add(self.glass_brain_actor)
        if plot_pial_surf:
            scene.add(self.pial_surf_l_actor)
            scene.add(self.pial_surf_r_actor)
        
        # Add actors
        if plot_points:
            stream_actor = actor.point(self.bundle.reshape(-1, 3), 
                                       colors=window.colors.coral, point_radius=3e-3)
        else:
            stream_actor = actor.line(self.bundle, 
                                      fake_tube=True, linewidth=6, colors=colors)
        scene.add(stream_actor)
        scene.ResetCamera() # don't remove this line
        
        # Show
        if interactive:
            window.show(scene, size=(1200,1200))
        
        # Save
        arr = window.snapshot(scene, size=(1200, 1200))
        if apply_autocrop:
            arr = autocrop(arr, border=5)

        if save_to is not None:
            if apply_autocrop:
                im = Image.fromarray(arr)
                im.save(save_to)
            else:
                window.record(scene, n_frames=1, reset_camera=False,
                            out_path=save_to, size=(1200, 1200))
        if do_plot:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.grid(False)
            ax.axis('off')
            ax.imshow(arr)
        if ret_img:
            return arr

    def visualize_bundle_with_metric(self, metric, **kwargs):
        '''Visualize bundle with metric mapped to each point'''
        values = self.get_metric(metric)
        colors = value2color(values.flatten(), label=metric.upper(), **kwargs)
        return self.visualize_bundle(colors, **kwargs)

    def visualize_bundle_with_disk(self, metric, indx=None, **kwargs):
        '''Visualize bundle with metric mapped to along-tract segments'''
        indx = self.assignment_map if indx is None else indx
        colors = value2diskcolor(self.bundle, metric, indx=indx, **kwargs)
        return self.visualize_bundle(colors=colors, **kwargs)