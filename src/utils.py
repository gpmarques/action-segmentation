""" utils module

This module implements utility methods to the proper working of this tool

...
Methods
-------
cluster_to_segment_bounds(clusters: pd.Series)
    Method that maps the cluster labels of segments of a video into a list of dicts
    encoding each label to each continuous segments that belong to that cluster
    label

positional_encoding(data: np.ndarray)
    Method that adds positional encoding to a numpy array
"""
import numpy as np
import math
import torch
import pandas as pd


def cluster_to_segment_bounds(clusters: pd.Series):
    """
    This method gets the cluster label of each segment and generates a list of dicts
    where each dict is composed of one key, a cluster label, and the corresponding
    value is a tuple with two elements, the start and end of a continuos segments
    that belong to that cluster.
    For example: [{1: (0, 23)}, {1: (34, 50)}, {2: (24, 34)}, {3: (50, 64)}]

    Parameters
    ----------
    clusters - pd.Series
        Pandas series of each segments cluster label

    Returns
    -------
    segs - list of dicts
        Each key is a label and a tuple is the start and end of continuous segments
        of that cluster label
    """

    segs = []
    for label in np.unique(clusters):
        idxs = np.where(clusters == label)[0]
        cont_start_idx = idxs[0]
        step = 0
        prev_idx = -1
        for cur_idx in idxs:
            if cur_idx != cont_start_idx + step:
                segs.append({label: (cont_start_idx, prev_idx + 1)})
                step = 1
                prev_idx = cur_idx
                cont_start_idx = cur_idx
            else:
                step += 1
                prev_idx = cur_idx

        if prev_idx + 1 < len(clusters) + 1:
            segs.append({label: (cont_start_idx, prev_idx + 1)})
    return segs


def positional_encoding(data: np.ndarray) -> np.ndarray:
    """
    Adds positional encoding in a numpy array. Positional encoding is a technique
    to inject order information into data, it was proposed in the paper Attention Is
    All you Need, for more info: https://arxiv.org/abs/1706.03762

    Parameters
    ----------
    data: np.ndarray
        data that the positional encoding will be added

    Returns
    -------
    np.ndarray
        The input + the positional encoding
    """
    d_model = data.shape[1]
    length = data.shape[0]

    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    den_exp = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                        -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * den_exp)
    pe[:, 1::2] = torch.cos(position.float() * den_exp)

    return data + pe.numpy()
