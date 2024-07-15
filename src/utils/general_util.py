import os
import logging
import random
import inspect
import torch
import numpy as np
import pytorch_lightning as pl


def getCurrentMemoryUsage():
    # From https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

    return int(memusage.strip())


def set_seed(seed=None, seed_torch=True):
    # Adapated from NMA-DL tutorials (https://deeplearning.neuromatch.io/)

    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    logging.info(f'Random seed {seed} has been set.')


def set_seed_pl(seed=None):
    if seed is None:
        seed = np.random.choice(2 ** 32)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # In case that `DataLoader` is used
    # Adapated from NMA-DL tutorials (https://deeplearning.neuromatch.io/)

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def set_device():
    # Adapated from NMA-DL tutorials (https://deeplearning.neuromatch.io/)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logging.info("GPU is not enabled in this notebook.")
    else:
        logging.info("GPU is enabled in this notebook.")
    return device


def make_dir(dirpath):
    if not os.path.exists(dirpath):
        original_umask = os.umask(0)
        try:
            os.makedirs(dirpath, mode=0o777)
            logging.info(f"Made directory at {dirpath}")
        finally:
            os.umask(original_umask)
    else:
        logging.info(f"Directory already exist at {dirpath}")


def print_source_code(func):
    print(inspect.getsource(func))


def map_list_with_dict(ls, d):
    '''Map list as key to dictionery values'''
    return list(map(d.get, ls))

def get_rng(rng=None):
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        rng = rng
    else:
        rng = np.random.default_rng(0)
    return rng

def random_order(length, rng=0):
    '''Generate random order given length (used for random shuffling)'''
    rng = get_rng(rng)
    return rng.permutation(length)


def random_shuffle(arr, rng=0):
    '''Random shuffle array'''
    new_idx = random_order(len(arr), rng)
    return arr[new_idx]


def random_select(data, n_sample=1, rng=0):
    '''
        Randomly select samples from data, returns index
        Example usage: data[random_select(data)]
    '''
    rng = get_rng(rng)
    return rng.choice(len(data), size=n_sample, replace=False)


def random_select_consecutive(length, n_sample=5, rng=0):
    '''
        Randomly select consecutive samples given length, returns index
    '''
    n_sample = min(length, n_sample)
    rng = get_rng(rng)
    idx = rng.choice(length-n_sample+1, size=1, replace=False)[0]
    return np.arange(idx, idx+n_sample)


def is_iter(obj):
    '''Check if an object is iterable (work for string, which is iterable)'''
    try:
        _ = iter(obj)
    except TypeError as te:
        return False
    return True