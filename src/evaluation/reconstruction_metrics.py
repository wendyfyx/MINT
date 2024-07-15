import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from core.BundleVisualizer import BundleVisualizer


def compute_metric(x, xb, metric='mae', feature_idx=None):
    if feature_idx is not None:
        x = x[:,:,feature_idx]
        xb = xb[:,:,feature_idx]
    if len(x.shape) > 2:
        x = x.reshape((x.shape[0],-1))
        xb = xb.reshape((xb.shape[0], -1))
    if metric.lower() == 'mae':
        return mean_absolute_error(x, xb)
    elif metric.lower() == 'mape':
        return mean_absolute_percentage_error(x, xb)
    elif metric.lower() == 'me':
        return (xb-x).mean()
    elif metric.lower() == 'mean_orig':
        return x.mean()
    elif metric.lower() == 'mean_recon':
        return xb.mean()
    elif metric.lower() == 'l2':
        return np.linalg.norm(x-xb)


def get_segment_metric(x, x_recon, bname, n_segments=100, segment_idx=None, feature_idx=None, metric='mae'):
    indx = BundleVisualizer(x, bundle_type=bname).get_assignment_map(n_segments=n_segments) if segment_idx is None else segment_idx
    scores = []
    if feature_idx is not None:
        x = x[:,:,feature_idx]
        x_recon = x_recon[:,:,feature_idx]
    n_features = x.shape[2]
    for i in range(n_segments):
        indx_segment = np.where(indx==i)
        segment = x.reshape(-1, n_features)[indx_segment]
        segment_recon = x_recon.reshape(-1, n_features)[indx_segment]
        if len(segment)==0 and len(segment_recon)==0:
            scores.append(0)
        else:
            scores.append(compute_metric(segment, segment_recon, metric=metric))
    return np.array(scores)
