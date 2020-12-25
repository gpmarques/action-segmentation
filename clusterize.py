from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering


class ClusterStrategy(ABC):

    cluster = NotImplemented
    strategy_name = NotImplemented
    path = NotImplemented
    random_state = 42

    @abstractmethod
    def clusterize(self, data: np.ndarray) -> None:
        pass

    def save_cluster_df(self,
                        data: np.ndarray,
                        clusters: np.ndarray) -> pd.DataFrame:
        """ Creates a dataframe with features with dimensions reduced by PCA
            and TSNE to visualise them

        args:
        features - 2d np array where each row is a feature of a segment
        clusters - cluster label of each row (segment)

        return:
        vis_feat_df - a pandas dataframe with 2 components representing the
                      features and another columns with the cluster labels
        """
        pca = PCA(n_components=50)
        red_data = pca.fit_transform(data)

        tsne = TSNE(n_components=2, perplexity=50)
        data_vis = tsne.fit_transform(red_data)

        vis_feat_df = pd.DataFrame(data_vis, columns=["x", "y"])
        vis_feat_df["cluster"] = clusters

        df_path = "_".join([self.path, self.strategy_name + ".feather"])
        vis_feat_df.to_feather(df_path)

        return vis_feat_df

    def get_cluster(self):
        df_path = "_".join([self.path, self.strategy_name + ".feather"])
        return pd.read_feather(df_path)


class KMeansClusterStrategy(ClusterStrategy):

    def __init__(self, k: int, path: str):
        self.path = path
        self.strategy_name = "kmeans" + "_" + str(k)
        self.cluster = KMeans(n_clusters=k,
                              random_state=self.random_state)

    def clusterize(self, data: np.ndarray) -> tuple:
        self.cluster.fit(data)
        labels = self.cluster.labels_
        df = self.save_cluster_df(data, labels)
        return labels, df


class GaussianMixtureClusterStrategy(ClusterStrategy):

    def __init__(self, n: int, path: str, cov_type: str = 'full'):
        self.strategy_name = "gm" + "_" + str(n)
        self.cluster = GaussianMixture(n_components=n,
                                       covariance_type=cov_type,
                                       random_state=self.random_state)

    def clusterize(self, data: np.ndarray) -> tuple:
        self.cluster.fit(data)
        labels = self.cluster.predict(data)
        df = self.save_cluster_df(data, labels)
        return labels, df


class SpectralClusterStrategy(ClusterStrategy):

    def __init__(self, n: int, path: str, affinity: str = 'rbf'):
        self.strategy_name = "sc" + "_" + str(n)
        self.cluster = SpectralClustering(n_clusters=n,
                                          affinity=affinity,
                                          random_state=self.random_state)

    def clusterize(self, data: np.ndarray) -> tuple:
        self.cluster.fit(data)
        labels = self.cluster.labels_
        df = self.save_cluster_df(data, labels)
        return labels, df
