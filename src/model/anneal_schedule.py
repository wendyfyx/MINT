import math 
import numpy as np

'''From https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb'''

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L  

def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L    

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L    

def frange(start, stop, step, n_epoch):
    L = np.ones(n_epoch)
    v , i = start , 0
    while v <= stop:
        L[i] = v
        v += step
        i += 1
    return L

def get_anneal_schedule(anneal_mode, n_step, n_cycle=1, ratio=0.5):
    '''Get the annealing schedule for the KL weight'''
    if "linear" in anneal_mode:
        return frange_cycle_linear(0, 1, n_step, n_cycle=n_cycle, ratio=ratio)
    elif "sigmoid" in anneal_mode:
        return frange_cycle_sigmoid(0, 1, n_step, n_cycle=n_cycle, ratio=ratio)
    elif "cosine" in anneal_mode:
        return frange_cycle_cosine(0, 1, n_step, n_cycle=n_cycle, ratio=ratio)
    else:
        return np.ones(n_step)