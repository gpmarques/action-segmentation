import unittest
import numpy as np

from clusters import KMeansClusterStrategy, AgglomerativeClusterStrategy
from clusters import ClusterFactory


class TestClusters(unittest.TestCase):

    def setUp(self):
        self.test = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1],
                              [2, 0, 0, 0],
                              [0, 2, 0, 0]])

    def test_kmeans_strategy(self):
        kmeans = KMeansClusterStrategy(n=4)
        labels = kmeans.estimate(self.test)
        assert len(np.unique(labels)) == 4
        assert labels[0] == labels[-2]
        assert labels[1] == labels[-1]

    def test_agglomerative_strategy(self):
        agglomerative = AgglomerativeClusterStrategy(n=4)
        labels = agglomerative.estimate(self.test)
        assert len(np.unique(labels)) == 4
        assert labels[0] == labels[-2]
        assert labels[1] == labels[-1]

    def test_auto_cluster(self):
        # setting not optimal number of cluster n = 2
        kmeans = KMeansClusterStrategy(n=2)
        labels = kmeans.auto(self.test)

        # optimal number of cluster must be n = 3, given self.test
        assert len(np.unique(labels)) == 3

    def test_cluster_factory(self):
        clusters = ClusterFactory.values_list()
        assert isinstance(ClusterFactory.get(clusters[0])(n=2),
                          KMeansClusterStrategy)
        assert isinstance(ClusterFactory.get(clusters[1])(n=2),
                          AgglomerativeClusterStrategy)


if __name__ == '__main__':
    unittest.main()
