import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from utils.general_util import map_list_with_dict


def autocrop(image, border=10):
    '''
        Automatically crop image to remove whole rows/columns with white space
        Adatped from https://stackoverflow.com/a/14211727
    '''
    image_data = np.asarray(image)
    image_data_bw = image_data.max(axis=2)
    non_empty_columns = np.where(image_data_bw.min(axis=0)<255)[0]
    non_empty_rows = np.where(image_data_bw.min(axis=1)<255)[0]
    cropBox = (min(non_empty_rows)-border, max(non_empty_rows)+border, 
               min(non_empty_columns)-border, max(non_empty_columns)+border)
    image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
    return image_data_new


def labeled_colormap(cmap_name, labels, plot_cmap=True, use_linspace=True):
    '''
        Make custom colormap with given labels
        Returns a dictionary of label name with color value 
    '''
    cmap = plt.get_cmap(cmap_name)
    if use_linspace:
        color_list = cmap(np.linspace(0, 1.0, len(labels)+1))
    else:
        color_list = cmap(np.arange(len(labels)+1))
    colormap = dict(zip(labels, color_list[:-1]))

    if plot_cmap:
        for i, (name, c) in enumerate(colormap.items()):
            plt.axhline(-i, linewidth=10, c=c, label=name)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

    return colormap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
        Generate custom range colormap from existing plt ones
    '''
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def symmetrical_colormap(cmap_settings, new_name = None ):
    ''' 
        This function take a colormap and create a new one, as the concatenation of itself by a symmetrical fold.
        From https://stackoverflow.com/a/67005578
    '''
    # get the colormap
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_"+cmap_settings[0]  # ex: 'sym_Blues'
    
    # this defined the roughness of the colormap, 128 fine
    n= 128 
    
    # get the list of color from colormap
    colors_r = cmap(np.linspace(0, 1, n))    # take the standard colormap # 'right-part'
    colors_l = colors_r[::-1]                # take the first list of color and flip the order # "left-part"

    # combine them and build a new colormap
    colors = np.vstack((colors_l, colors_r))
    return mcolors.LinearSegmentedColormap.from_list(new_name, colors)