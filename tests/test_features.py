import unittest
from features import Features
import numpy as np
import os
import shutil


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.video_path = "./tests/assets/RoadAccidents010_x264.mp4"
        self.features = np.zeros((2, 1024))
        self.features_obj = Features(self.video_path, "SlowFast")

    def test_path(self):
        assert self.features_obj.path == self.video_path + "_features/SlowFast"

    def test_read_n_write(self):
        self.features_obj.write(ith=1, features=self.features)
        files = os.listdir(self.features_obj.path)
        assert files[0] == "0001_features.npy"

        read_feat = self.features_obj()
        assert read_feat.shape == self.features.shape

    def tearDown(self):
        shutil.rmtree(self.video_path + "_features")


if __name__ == '__main__':
    unittest.main()
