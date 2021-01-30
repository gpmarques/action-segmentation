from abc import ABC, abstractmethod
import os

import streamlit as st
import altair as alt

from clusters import ClusterFactory
from features import Features
from positional_encoding import PositionalEncoding
from extractors import FeatureExtraction, SlowFastStrategy

import plots
import utils


class Page(ABC):
    """ Class that defines the methods that a page must have """

    @abstractmethod
    def _init_layout(self):
        """ Abstract method that must be implemented in concrete implementations
        of this class. It is expected to layout all components that should go
        into a page.
        """
        pass

    def run(self):
        self._init_layout()


class VideoSegmentationCluster(Page):

    folder_path = "./data"

    def _init_layout(self):
        self.folder_selection_component()

        if self.folder_path:
            self.file_selection_component()
            self.feature_extractor_component()
            self.cluster_selection_component()
            st.video(self.video_path)
            self.cluster_plot_component()

    @staticmethod
    @st.cache(allow_output_mutation=True)
    def get_feature_extractor():
        """ Static method that loads the Feature Extractor and stores it in a
        cache to avoid reloading
        """
        feature_extractor = FeatureExtraction()
        feature_extractor.extraction_strategy = SlowFastStrategy()
        return feature_extractor

    def folder_selection_component(self):
        self.change_dir_checkbox = st.sidebar.checkbox("Change directory")
        if self.change_dir_checkbox:
            self.folder_path = st.text_input("Enter the folder path where your videos are")

    def file_selection_component(self):
        """ UI file selection component """
        filenames = os.listdir(self.folder_path)  # list all files
        filenames = [  # Filter only mp4 file names
            filename for filename in filenames
            if filename.split(".")[-1] in ["mp4"]]
        st.sidebar.title("Pick your video")
        self.selected_filename = st.sidebar.selectbox('Select a file',
                                                      filenames)
        self.video_path = os.path.join(self.folder_path,
                                       self.selected_filename)

    def feature_extractor_component(self):
        """ UI file selection component """
        self.feat_io = Features(self.video_path)
        self.feature_extractor = VideoSegmentationCluster.get_feature_extractor()

        if self.feat_io.has_features is False:
            with st.spinner("Extracting..."):
                progress_bar = st.progress(0)
                for i in self.feature_extractor.extract(self.video_path):
                    progress_bar.progress(i)
                progress_bar.empty()

    def cluster_selection_component(self):
        """ UI cluster strategy component """
        st.sidebar.title("Cluster information")

        available_cluster_strategies = ClusterFactory.values_list()
        self.selected_cluster = st.sidebar.selectbox('Select a cluster strategy',
                                                     available_cluster_strategies)

        self.num_cluster_slider = st.sidebar.slider("Select number of clusters",
                                                    min_value=2, max_value=8)
        self.positional_encoding_checkbox = st.sidebar.checkbox("Use Positional Encoding")
        self.auto_cluster_checkbox = st.sidebar.checkbox("Automatic find the optimal number of clusters")

        self.cluster = ClusterFactory.get(self.selected_cluster)(n=self.num_cluster_slider)

        if self.positional_encoding_checkbox:
            self.features = PositionalEncoding()(self.feat_io.read(preproc=False))
        else:
            self.features = self.feat_io.read(preproc=False)

        if self.auto_cluster_checkbox:
            self.cluster_labels = self.cluster.auto(self.features)
        else:
            self.cluster_labels = self.cluster.estimate(self.features)

    def cluster_plot_component(self):
        self.cluster_df = self.cluster.df(self.features, self.cluster_labels)
        timeline_chart = plots.video_timeline(utils.cluster_to_segment_bounds(self.cluster_df.labels))
        scatter_chart = plots.clusters(self.cluster_df)

        st.altair_chart(alt.vconcat(timeline_chart, scatter_chart))
