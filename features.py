""" Features module

This module implements all features related classes and methods

...
Classes
-------
Features
    Class used to represent and manage the features extracted of a video by some
    feature extractor

"""
from data_io import Npy
import numpy as np
import os


class Features:
    """ Class that represents the features extracted from a video

    Attributes
    ----------
    _path: str
        The path to the video of this features

    path: str
        The path where this features are stored

    _extractor: str
        The name of the extractor that extracted this features

    has_features: bool
        Indicates whether this features have already been extracted

    npy_io: Npy
        Object that manages the read and write functionalities of npy files,
        which is the file type that features are stored

    FEATURE_BASE_FNAME: str
        All npy files that store features have to have this suffix

    Methods
    ------

    __call__()
        Reads the path attribute with npy_io and returns a numpy array

    write(ith: int, features: np.ndarray)
        Writes features to path with the prefix XXXX with ith
    """

    # suffix of all features written by this module
    FEATURE_BASE_FNAME = "features.npy"

    def __init__(self, path: str, extractor: str):
        """
        Parameters
        ----------
        path: str
            path to the video for the features extracted

        str: str
            The name of the extractor that extracted this features

        npy_io: Npy
            Object responsible for all i/o operations with npy files
        """
        self._path = path
        self._extractor = extractor
        self.npy_io = Npy()

    @property
    def path(self):
        """ Path to this features """
        if os.path.isdir(self._path+"_features") is False:
            os.mkdir(self._path+"_features")
        return os.path.join(self._path+"_features", self._extractor)

    @property
    def has_features(self):
        """ Checks if this features exist """
        features_count = self.npy_io.count_npy_files(self.path)
        return features_count > 0

    def __call__(self) -> np.ndarray:
        """ Reads this features with the npy_io object in path """
        return self.npy_io.read(self.path)

    def write(self, ith: int, features: np.ndarray):
        """ Writes features with ith prefix

        It first checks is the path where this features should be stored exist
        if it doesn't, creates it. Then creates the file name of the feature
        passed, which is of the form XXXX_features.npy.

        Parameters
        ----------
        ith: int
            The prefix of a feature filename

        features: np.ndarray
            Numpy array storing a feature
        """
        if os.path.isdir(self.path) is False:
            os.mkdir(self.path)

        fname = f"{ith:04}_{self.FEATURE_BASE_FNAME}"
        self.npy_io.write(self.path, fname, features)
