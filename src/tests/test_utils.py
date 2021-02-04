import unittest
from utils import cluster_to_segment_bounds, positional_encoding
import pandas as pd
import numpy as np


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.clusters = pd.DataFrame({"labels": [0, 0, 1, 1, 1, 0, 0, 2, 2]})
        self.features = np.ones((1, 4))

    def test_cluster_to_segment_bounds(self):
        segs = cluster_to_segment_bounds(self.clusters.labels)
        expected_segs = [{0: (0, 2)}, {0: (5, 7)},
                         {1: (2, 5)}, {2: (7, 9)}]
        assert segs == expected_segs

    def test_positional_encoding(self):
        pos_encoded_feat = positional_encoding(self.features)
        assert np.all(pos_encoded_feat[:, 0::2] == 1.0)
        assert np.all(pos_encoded_feat[:, 1::2] == 2.0)


if __name__ == '__main__':
    unittest.main()
