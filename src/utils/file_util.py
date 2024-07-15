import logging
import numpy as np
import json
import h5py
import yaml
import pickle
import os.path


def file_exists(f):
    '''Check if file exists'''
    return os.path.isfile(f)


def make_directory(dir):
    '''Make directory if it does not already exist'''
    if not os.path.exists(dir):
        os.makedirs(dir)
        logging.info(f"Created directory {dir}.")


def load_list(fname):
    '''Load list from file (txt)'''
    with open(fname, 'r') as f:
        lines = [line.rstrip() for line in f]
        logging.info(f"Loaded list (N={len(lines)}) from {fname}.")
        return lines
    

def save_list(ls, fname):
    '''Load list from file (txt)'''
    with open(fname, 'w') as f:
        for line in ls:
            f.write(f"{line}\n")
        logging.info(f"Saved list (N={len(ls)}) to {fname}.")


def load_yaml(fname):
    '''Load data from yaml file'''
    with open(fname, "r") as stream:
        try:
            logging.info(f"Loading yaml data from {fname}.")
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.warning(f"Something is wrong when loading yaml from {fname}!")
            return exc


def save_json(data, fname):
    '''Save data to json file'''
    with open(fname, 'w') as fp:
        json.dump(data, fp)
        logging.info(f"Saved json data to {fname}.")


def load_json(fname):
    '''Load data from json file'''
    with open(fname, 'r') as f:
        logging.info(f"Loading json data from {fname}.")
        return json.load(f)


def save_pickle(data, fname):
    '''Save data to pickle file'''
    with open(fname, 'wb') as f:
        pickle.dump(data, f, protocol=4)
        logging.info(f"Saved pickle data to {fname}.")


def load_pickle(fname):
    '''Load data from pickle file'''
    try:
        with open(fname, 'rb') as f:
            logging.info(f"Loading pickle data from {fname}.")
            return pickle.load(f)
    except FileNotFoundError:
        logging.info(f"Wrong pickle file path or file does not exist at {fname}.")


def save_dict_to_h5(data, h5_path, mode='a', overwrite=False, **kwargs):
    '''Save dictionary as dataset to h5 file'''
    with h5py.File(h5_path, mode) as hf:
        for k, v in data.items():
            if k in hf:
                logging.info(f"{k} already exists (overwite is set to {overwrite}).")
                if overwrite: # key exists and do overwrite
                    del hf[k]
                else: # key exists but do not overwrite
                    return
            hf.create_dataset(k, data=v, **kwargs)
            logging.debug(f"Saved data {k} {v.shape if isinstance(v, np.ndarray) else v} to {h5_path}.")
        logging.info(f"Saved dict with keys {data.keys()} to {h5_path}.")


def save_attr_to_h5(data, h5_path, mode='a', overwrite=False, base_key=None, **kwargs):
    '''Save dictionary as attribute to h5 file at location specified by base_key'''
    with h5py.File(h5_path, mode) as hf:
        grp = hf if base_key is None else hf[base_key]
        location = 'root' if base_key is None else base_key
        for k, v in data.items():
            if k in hf:
                logging.info(f"{k} already exists at [{location}] (overwite is set to {overwrite}).")
                if not overwrite: # key exists and do not overwrite
                    return
            grp.attrs[k] = v
            logging.info(f"Saved attribute ({k}, {v}) to {h5_path} at [{location}].")


def load_data_from_h5(h5_path, key):
    '''Load dataset from h5 file'''
    with h5py.File(h5_path, 'r') as hf: 
        if key not in hf:
            logging.warning(f"{key} does not exist in h5_path")
            return
        return hf[key][...]
    
    
def key_in_h5(h5_path, key):
    '''Check if key is in .h5 file'''
    with h5py.File(h5_path, 'r') as hf: 
        return key in hf
