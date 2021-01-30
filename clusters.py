from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
from enum import Enum

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score

import config


class ClusterStrategy(ABC):

    estimator = NotImplemented  # variable that will store the cluster object
    random_state = 42

    @abstractmethod
    def estimate(self, data: np.ndarray) -> None:
        """ Abstract method that must be implemented in concrete implementations
        of this class. It is expected to perform a clustering process in a da-
        -taset

        args:
        data - numpy array storing the data to be clustered
        """
        pass

    @abstractmethod
    def set_estimator(self, n: int) -> None:
        pass

    def df(self, data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """ Creates a dataframe by first reducing the dimensionality of the
        features with PCA and TSNE to enable visualisation

        args:
        features - 2d np.array of shape (num_segments, feature_len)
        labels - cluster label of each segment

        return:
        vis_feat_df - a pandas dataframe with 2 components representing the
                      features and another column with cluster labels
        """
        if data.shape[0] > 50:
            pca = PCA(n_components=50)
            data = pca.fit_transform(data)

        tsne = TSNE(n_components=2, perplexity=50)
        data_vis = tsne.fit_transform(data)

        vis_feat_df = pd.DataFrame(data_vis, columns=["x", "y"])
        vis_feat_df["labels"] = labels

        return vis_feat_df

    def auto(self, data: np.ndarray) -> np.ndarray:
        sil_vals = []
        max_sil = -math.inf
        max_sil_estimator = None
        max_sil_labels = None
        not_inc_iter = 0
        upper_n_cluster_lim = int(0.5 * data.shape[0] / config.SEGMENT_LENGTH)

        for n in range(3, upper_n_cluster_lim):
            self.set_estimator(n)
            labels = self.estimate(data)
            sil_vals.append(silhouette_score(data, labels))

            if (max_sil < 0 and sil_vals[-1] > max_sil) or\
               (sil_vals[-1] > max_sil):
                max_sil = sil_vals[-1]
                max_sil_estimator = self.estimator
                max_sil_labels = labels
                not_inc_iter = 0
            else:
                not_inc_iter += 1

            print(n, not_inc_iter, max_sil)

            if not_inc_iter >= 2:
                break

        self.estimator = max_sil_estimator

        return max_sil_labels


class KMeansClusterStrategy(ClusterStrategy):

    def __init__(self, n: int):
        self.set_estimator(n)

    def estimate(self, data: np.ndarray) -> np.ndarray:
        self.estimator.fit(data)
        labels = self.estimator.labels_
        return labels

    def set_estimator(self, n):
        self.estimator = KMeans(n_clusters=n,
                                random_state=self.random_state)


class GaussianMixtureClusterStrategy(ClusterStrategy):

    def __init__(self, n: int):
        self.set_estimator(n)

    def estimate(self, data: np.ndarray) -> np.ndarray:
        self.estimator.fit(data)
        labels = self.estimator.predict(data)
        return labels

    def set_estimator(self, n):
        self.estimator = GaussianMixture(n_components=n,
                                         random_state=self.random_state)


class BayesMixtureClusterStrategy(ClusterStrategy):

    def __init__(self, n: int):
        self.set_estimator(n)

    def estimate(self, data: np.ndarray) -> np.ndarray:
        self.estimator.fit(data)
        return self.estimator.predict(data)

    def set_estimator(self, n: int) -> None:
        self.estimator = BayesianGaussianMixture(n_components=n,
                                                 random_state=self.random_state)

    def auto(self, data: np.ndarray) -> np.ndarray:
        upper_clust_lim = int(data.shape[0] / 16 * 0.5)
        self.set_estimator(upper_clust_lim)
        return self.estimate(data)


class AgglomerativeClusterStrategy(ClusterStrategy):

    def __init__(self, n: int,
                 linkage: str = "average", affinity: str = 'cosine'):
        self._linkage = linkage
        self._affinity = affinity
        self.set_estimator(n)

    def estimate(self, data: np.ndarray) -> np.ndarray:
        self.estimator.fit(data)
        return self.estimator.labels_

    def set_estimator(self, n: int):
        self.estimator = AgglomerativeClustering(n_clusters=n,
                                                 linkage=self._linkage,
                                                 affinity=self._affinity)


class ClusterFactory(Enum):

    KMEANS = "KMeans"
    GAUSSIANMIXTURE = "Gaussian Mixture"
    BAYESMIXTURE = "Bayesian Gaussian Mixture"
    HIERARCHICAL = "Agglomerative"

    @staticmethod
    def values_list() -> list:
        return [ctype.value for ctype in ClusterFactory]

    @staticmethod
    def get(cluster_type: str) -> ClusterStrategy:
        return {
            ClusterFactory.KMEANS.value: KMeansClusterStrategy,
            ClusterFactory.GAUSSIANMIXTURE.value: GaussianMixtureClusterStrategy,
            ClusterFactory.BAYESMIXTURE.value: BayesMixtureClusterStrategy,
            ClusterFactory.HIERARCHICAL.value: AgglomerativeClusterStrategy
        }[cluster_type]
