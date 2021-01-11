from data_io import Npy

import numpy as np
from sklearn.preprocessing import scale


class Features:

    # suffix of all features written by this module
    FEATURE_BASE_FNAME = "feature.npy"

    def __init__(self, path: str):
        """
        path: path to the video for the features extracted
        npy_io: object responsible for all i/o operations with npy files
        """
        self._path = path
        self.npy_io = Npy()

    @property
    def path(self):
        return self._path+"_features"

    @property
    def has_features(self):
        features_count = self.npy_io.count_npy_files(self.path)
        return features_count > 0

    def read(self, preproc=True):
        feat = self.npy_io.read(self.path)
        feat = feat.reshape((feat.shape[0] * feat.shape[1], feat.shape[2]))

        return scale(feat) if preproc else feat

    def write(self, ith: int, feature: np.array):
        fname = self._build_feat_fname(ith)
        self.__assert_has_prefix_index(fname)
        self.npy_io.write(self.path, fname, feature)

    def _build_feat_fname(self, i: int):
        return f"{i:04}_{self.FEATURE_BASE_FNAME}"

    def __assert_has_prefix_index(self, fname: str):
        assert fname.split("_")[0].isdigit()
        return True
