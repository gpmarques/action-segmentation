""" page module

This module implements all pages related classes and methods. A page here is a
streamlit script encapsulated by a class.

...
Classes
-------
Page
    Abstract class used to represent a generic page to be built with streamlit
    components

VideoSegmentationCluster
    Concrete class used to represent the main page of this video segmentation tool
    with clusters
"""
from abc import ABC, abstractmethod
import os

import streamlit as st
import altair as alt

from clusters import ClusterFactory
from video import Video
from extractors import ExtractorFactory
import plots
from utils import cluster_to_segment_bounds, positional_encoding
from s3 import S3


class Page(ABC):
    """
    Abstract class that represents a generic page to be built with streamlit
    components

    Methods
    -------
    _init_page(self)
        Subclasses of Page must implement this method. The implementation of this
        method must have the layout of a page designed with the streamlit framework

    run(self)
        Method that calls the method _init_layout and starts a streamlit page

    """

    @abstractmethod
    def _init_page(self):
        """ Abstract method that must be implemented in concrete implementations
        of this class. It is expected to layout all components that should go
        into a page.
        """
        pass

    def run(self):
        """ Method that calls the method _init_layout and starts a streamlit page """
        self._init_page()


class VideoSegmentationCluster(Page):
    """
    Class that represents a the streamlit page of the video segmentation cluster
    tool

    Parameters
    ----------
    folder_path: str
        Path to the folder where the videos a user might want to segment are

    change_dir_checkbox: bool
        Boolean that flags if the user ticked a checkbox UI component to change
        the folder_path

    selected_filename: str
        Video's file name selected by the user

    video_path: str
        Path to the selected video, it is the folder_path + selected_filename

    selected_extractor: str
        Name of the feature extractor selected by the user

    feature_extractor: ExtractionStrategy
        ExtractionStrategy object representing the selected feature extractor

    video: Video
        Video object representing the video selected by the user

    selected_cluster: str
        Name of the cluster strategy selected by the user

    positional_encoding_checkbox: bool
        Boolean that flags if the user ticked a checkbox UI component to use
        positional encoding

    auto_cluster_checkbox: bool
        Boolean that flags if the user ticked a checkbox UI component to use
        the automatic optimal number of clusters algorithm

    num_cluster_slider: int
        Number of clusters the user wants to cluster the video

    cluster: ClusterStrategy
        ClusterStrategy object representing the selected cluster algorithm

    features: np.ndarray
        Numpy array of shape (S, F) with the features of the selected video, where
        S is the number of segments features this video has been divided into and
        F is the dimensionality of each feature

    cluster_labels: np.ndarray
        Cluster label of each feature segment

    cluster_df: pd.DataFrame
        Dataframe with the features after having its dimensionality reduced and
        its cluster labels
    """

    folder_path = "./data"
    bucket_name = "segmentation-videos"
    change_dir_checkbox = None
    selected_filename = None
    video_path = None
    selected_extractor = None
    feature_extractor = None
    video = None
    selected_cluster = None
    positional_encoding_checkbox = None
    auto_cluster_checkbox = None
    num_cluster_slider = None
    cluster = None
    features = None
    cluster_labels = None
    cluster_df = None

    def _init_page(self):
        """ This method calls all component methods in the correct order that
        they appear in the UI and manages any logical dependency between them.
        """

        if self.folder_path:
            self._file_selection_component()
            if self.video_path:
                self._feature_extractor_component()
                self._cluster_selection_component()
                if self.video_path.split(".")[-1] == 'avi':
                    print(self.video_path)
                    st.video(self.video_path, format='video/avi')
                else:
                    st.video(self.video_path)
                self._cluster_plot_component()

    def _file_selection_component(self):
        """
        Method that creates file selection UI components and their logical
        dependencies.
        """
        self.s3 = S3()
        filenames = self.s3.list_objects(bucket_name=self.bucket_name,
                                         prefix="videos",
                                         file_extension=['mp4', 'avi'])

        filenames = [  # removing the videos path from the filename
            filename.split("/")[-1] for filename in filenames]
        print(filenames)

        # Creates a title UI component
        st.sidebar.title("Pick your video")
        # creates the selection box component UI and stores the selected
        # file
        self.selected_filename = st.sidebar.selectbox('Select a file',
                                                      filenames)
        self.video_path = os.path.join(self.folder_path,
                                       self.selected_filename)

        if os.path.exists(self.video_path) is False:
            self.s3.download_object(self.bucket_name,
                                    os.path.join('videos', self.selected_filename),
                                    self.folder_path)

    def _feature_extractor_component(self):
        """
        Method that creates all feature extraction UI components and their logical
        dependencies.
        """
        # creates a title UI component
        st.sidebar.title("Pick a feature extractor")

        # gets all available extraction strategies
        available_extraction_strategies = ExtractorFactory.values_list()

        # creates a selection box UI component with the available_extraction_strategies
        # and stores the selected in a variable
        self.selected_extractor = st.sidebar.selectbox('Select an extraction strategy',
                                                       available_extraction_strategies)

        # gets the feature extractor object from the ExtractorFactory
        self.feature_extractor = ExtractorFactory.get(self.selected_extractor)()

        # creates the video object with its path and selected extractor name
        self.video = Video(self.video_path, self.selected_extractor)

        # if the video doesn't have features extracted with this extractor
        # it extracts the features
        if self.video.features.has_features is False:
            with st.spinner("Extracting..."):
                self.feature_extractor.extract(self.video)

    def _cluster_selection_component(self):
        """
        Method that creates the cluster selection UI components and their logical
        dependencies.
        """
        # creates a title UI component
        st.sidebar.title("Cluster information")

        # gets all available cluster strategies
        available_cluster_strategies = ClusterFactory.values_list()

        # creates a selection box UI component with the available_cluster_strategies
        # and stores the selected cluster strategy in a variable
        self.selected_cluster = st.sidebar.selectbox('Select a cluster strategy',
                                                     available_cluster_strategies)

        # creates the positional encoding checkbox
        self.positional_encoding_checkbox = st.sidebar.checkbox("Use Positional Encoding")
        # creates the auto cluster checkbox
        self.auto_cluster_checkbox = st.sidebar.checkbox("Automatic find the optimal number of clusters")

        # if auto cluster is ticked, just initialize it with an integer
        if self.auto_cluster_checkbox:
            self.num_cluster_slider = 2
        else:  # otherwise creates a slider UI component so the user can pick the number of clusters
            self.num_cluster_slider = st.sidebar.slider("Select number of clusters",
                                                        min_value=2, max_value=8)

        # gets the feature extractor object from the ExtractorFactory
        self.cluster = ClusterFactory.get(self.selected_cluster)(n=self.num_cluster_slider)

        # gets the features from the video object
        self.features = self.video.features()

        # if the positional encoding checkbox is ticked, preprocess the features
        # with the positional_encoding function
        if self.positional_encoding_checkbox:
            self.features = positional_encoding(self.features)

        # if auto cluster checkbox is ticked, use the auto method
        if self.auto_cluster_checkbox:
            self.cluster_labels = self.cluster.auto(self.features)
        else:  # otherwise the estimate method
            self.cluster_labels = self.cluster.estimate(self.features)

    def _cluster_plot_component(self):
        """
        Method that creates the cluster plot UI components and their logical
        dependencies.
        """
        # creates the features and cluster labels dataframe
        self.cluster_df = self.cluster.df(self.features, self.cluster_labels)

        # creates the video timeline chart and scatter chart
        timeline_chart = plots.video_timeline(
            cluster_to_segment_bounds(self.cluster_df.labels))
        scatter_chart = plots.clusters(self.cluster_df)

        # creates the altair chart UI component
        st.altair_chart(alt.vconcat(timeline_chart, scatter_chart))
