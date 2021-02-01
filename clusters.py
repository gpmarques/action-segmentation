""" Clusters module

This module implements all cluster related classes and methods

...
Classes
-------
ClusterStrategy
    Abstract class used to represent a generic cluster strategy

KMeansClusterStrategy
    Concrete class used to represent the KMeans clustering strategy

AgglomerativeClusterStrategy
    Concrete class used to represent the Agglomerative clustering strategy

ClusterFactory
    Class used to represent a cluster factory, it contains the cluster strategies
    available and it makes them easily accessible

"""
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
from enum import Enum

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


class ClusterStrategy(ABC):
    """
    A class used to represent an abstract cluster algorithm strategy

    ...
    Attributes
    ----------
    estimator: object
        object with clustering capabilities
    random_state: int
        integer to make algorithms with random procedures be reproducible

    Methods
    -------
    estimate(data: np.ndarray)
        abstract method that must be implemented by subclasses of ClusterStrategy
        that is supposed to perform clustering

    set_estimator(n: int)
        abstract method that must be implemented by subclasses of ClusterStrategy
        that is supposed to initialize the estimator attribute

    df(data: np.nadarray, labels: np.ndarray)
        creates a dataframe with the data and labels, but first it reduces the
        dimensionality of data

    auto(data: np.ndarray)
        Finds the optimal numbers of cluster based on the sillhoute measure of
        different number of clusters
    """

    estimator = NotImplemented  # variable that will store the cluster object
    random_state = 42

    @abstractmethod
    def estimate(self, data: np.ndarray) -> None:
        """ Abstract method that must be implemented in concrete implementations
        of this class. It is expected to perform a clustering process in a da-
        -taset with the estimator attribute

        Parameters
        ----------
        data: np.ndarray
            2d numpy array with the data to be clustered
        """
        pass

    @abstractmethod
    def set_estimator(self, n: int) -> None:
        """ Abstract method that must be implemented in concrete implementations
        of this class. It is expected to initialize the estimator with an object
        with clustering capabilities

        Parameters
        ----------
        n: int
            number of clusters this estimator is supposed to find
        """
        pass

    def df(self, data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """ Creates a dataframe by first reducing the dimensionality of the
        features with PCA and TSNE to enable visualisation

        Parameters
        ----------
        features: np.ndarray
            2d np.array of shape (num_segments, feature_len)
        labels: np.ndarray
            1d np.array with the same number of rows as the features parameter
            that contains the cluster label of each segment

        Returns
        -------
        vis_feat_df
            a pandas dataframe with 2 components representing the features and
            another column with cluster labels
        """

        assert data.shape[0] == labels.shape[0]

        if data.shape[0] > 50:
            pca = PCA(n_components=50)
            data = pca.fit_transform(data)

        tsne = TSNE(n_components=2, perplexity=50)
        data_vis = tsne.fit_transform(data)

        vis_feat_df = pd.DataFrame(data_vis, columns=["x", "y"])
        vis_feat_df["labels"] = labels

        return vis_feat_df

    def auto(self, data: np.ndarray) -> np.ndarray:
        """ Finds the optimal numbers of cluster based on the sillhoute measure
            of different number of clusters

        Parameters
        ----------
        features - 2d np.array of shape (num_segments, feature_len)
        labels - cluster label of each segment

        Returns
        -------
        max_sil_labels
            The cluster labels of the optimal number of clusters
        """
        sil_vals = []
        max_sil = -math.inf
        max_sil_estimator = None
        max_sil_labels = None
        not_inc_iter = 0

        for n in range(3, 8):
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

            if not_inc_iter >= 2:
                break

        self.estimator = max_sil_estimator

        return max_sil_labels


class KMeansClusterStrategy(ClusterStrategy):
    """
    Class used to represent the KMeans clustering algorithm

    ...
    Attributes
    ----------
    estimator: KMeans scikit learn cluster object
        Object that performs the KMeans clustering algorith
    random_state: int
        integer to make algorithms with random procedures be reproducible

    Methods
    -------
    estimate(data: np.ndarray)
        It estimates the clusters of the data passed with the KMeans algorithm

    set_estimator(n: int)
        It initializes the estimator object with the KMeans scikit learn cluster
        object

    df(data: np.nadarray, labels: np.ndarray)
        creates a dataframe with the data and labels, but first it reduces the
        dimensionality of data

    auto(data: np.ndarray)
        Finds the optimal numbers of cluster based on the sillhoute measure of
        different number of clusters
    """

    def __init__(self, n: int):
        """
        Parameters
        ----------
        n: int
            The number of clusters the estimator object should generate
        """
        self.set_estimator(n)

    def estimate(self, data: np.ndarray) -> np.ndarray:
        """ It estimates the cluster labels for each sample in data

        Parameters
        ----------
        data: np.ndarray
            2d numpy array with the data to be clustered

        Returns
        -------
        labels: np.ndarray
            1d numpy array with the cluster label of each sample in data
        """
        self.estimator.fit(data)
        labels = self.estimator.labels_
        return labels

    def set_estimator(self, n):
        """ It initializes the estimator attribute with the KMeans scikit learn
        cluster object

        Parameters
        ----------
        n: int
            The number of clusters the estimator object should generate
        """
        self.estimator = KMeans(n_clusters=n,
                                random_state=self.random_state)


class AgglomerativeClusterStrategy(ClusterStrategy):
    """
    Class used to represent the Agglomerative clustering algorithm

    ...
    Attributes
    ----------
    estimator: AgglomerativeClustering scikit learn cluster object
        Object that performs the agglomerative clustering algorithm
    random_state: int
        integer to make algorithms with random procedures be reproducible

    Methods
    -------
    estimate(data: np.ndarray)
        It estimates the clusters of the data passed with the
        agglomerative clustering algorithm

    set_estimator(n: int)
        It initializes the estimator object with the AgglomerativeClustering
        scikit learn cluster object

    df(data: np.nadarray, labels: np.ndarray)
        creates a dataframe with the data and labels, but first it reduces the
        dimensionality of data

    auto(data: np.ndarray)
        Finds the optimal numbers of cluster based on the sillhoute measure of
        different number of clusters
    """

    def __init__(self, n: int,
                 linkage: str = "average", affinity: str = 'cosine'):
        """
        Parameters
        ----------
        n: int
            The number of clusters the estimator object should generate

        linkage: str, default = 'average'
            Which linkage criterion to use. The linkage criterion determines
            which distance to use between sets of observation. The algorithm
            will merge the pairs of cluster that minimize this criterion

        affinity: str, default = 'cosine'
            Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”,
            “manhattan”, “cosine”, or “precomputed”. If linkage is “ward”,
            only “euclidean” is accepted. If “precomputed”, a distance matrix
            (instead of a similarity matrix) is needed as input for the fit
            method.
        """
        self._linkage = linkage
        self._affinity = affinity
        self.set_estimator(n)

    def estimate(self, data: np.ndarray) -> np.ndarray:
        """ It estimates the cluster labels for each sample in data

        Parameters
        ----------
        data: np.ndarray
            2d numpy array with the data to be clustered

        Returns
        -------
        labels: np.ndarray
            1d numpy array with the cluster label of each sample in data
        """
        self.estimator.fit(data)
        return self.estimator.labels_

    def set_estimator(self, n: int):
        """ It initializes the estimator attribute with the AgglomerativeClustering
        scikit learn cluster object

        Parameters
        ----------
        n: int
            The number of clusters the estimator object should generate
        """
        self.estimator = AgglomerativeClustering(n_clusters=n,
                                                 linkage=self._linkage,
                                                 affinity=self._affinity)


class ClusterFactory(Enum):
    """
    Class used to represent a cluster strategy factory

    Attributes
    ----------
    KMEANS: enum member
        Enum member representing the KMeansClusterStrategy class

    HIERARCHICAL: enum member
        Enum member representing the AgglomerativeClusterStrategy class

    Methods
    -------
    values_list()
        Returns a list with each enum member value

    get(cluster_type: str)
        Returns the ClusterStrategy object matching the cluster_type parameter
        passed
    """

    KMEANS = "KMeans"
    HIERARCHICAL = "Agglomerative"

    @staticmethod
    def values_list() -> list:
        """ Returns a list with each enum member value """
        return [ctype.value for ctype in ClusterFactory]

    @staticmethod
    def get(cluster_type: str) -> ClusterStrategy:
        """ Returns the ClusterStrategy object matching the cluster_type parameter
        passed

        Parameters
        ----------
        cluster_type: str
            String with the value of some enum member of ClusterFactory

        Returns
        -------
        ClusterStrategy
            Returns the ClusterStrategy class of the cluster_type parameter
        """
        return {
            ClusterFactory.KMEANS.value: KMeansClusterStrategy,
            ClusterFactory.HIERARCHICAL.value: AgglomerativeClusterStrategy
        }[cluster_type]
