from data_io import Npy
import numpy as np
import os


class Features:

    # suffix of all features written by this module
    FEATURE_BASE_FNAME = "features.npy"

    def __init__(self, path: str, extractor: str):
        """
        path: path to the video for the features extracted
        str: extractor name
        npy_io: object responsible for all i/o operations with npy files
        """
        self._path = path
        self._extractor = extractor
        self.npy_io = Npy()

    @property
    def path(self):
        if os.path.isdir(self._path+"_features") is False:
            os.mkdir(self._path+"_features")
        return os.path.join(self._path+"_features", self._extractor)

    @property
    def has_features(self):
        features_count = self.npy_io.count_npy_files(self.path)
        return features_count > 0

    def __call__(self):
        return self.npy_io.read(self.path)

    def write(self, ith: int, features: np.array):
        if os.path.isdir(self.path) is False:
            os.mkdir(self.path)

        fname = f"{ith:04}_{self.FEATURE_BASE_FNAME}"
        self.npy_io.write(self.path, fname, features)
