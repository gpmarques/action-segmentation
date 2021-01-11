import streamlit as st
import altair as alt
import os

from extractors import FeatureExtraction, SlowFastStrategy
from clusters import ClusterFactory
from features import Features
from positional_encoding import PositionalEncoding
from video import Video
from segment_video import ActionSegmentation
import plots
import utils


st.title("Action Clusterization app")


@st.cache(allow_output_mutation=True)
def get_feature_extractor():
    feature_extractor = FeatureExtraction()
    feature_extractor.extraction_strategy = SlowFastStrategy()
    return feature_extractor


def file_selector(folder_path):
    """ Interface file selection component """
    filenames = os.listdir(folder_path)
    filenames = [
        filename for filename in filenames
        if filename.split(".")[-1] in ["mp4"]]

    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


feature_extractor = get_feature_extractor()
folder_path = './data'

if st.checkbox('Change directory'):
    folder_path = st.text_input('Enter folder path')

if folder_path:
    video_path = file_selector(folder_path=folder_path)
    feat_io = Features(video_path)
    video = Video(video_path)
    segmentator = ActionSegmentation(video_path)
    positional_encoding = PositionalEncoding()
    num_cluster_slider = 2
    metadata = video.metadata

    cluster = None
    c_labels = None

    pos_enc = st.sidebar.checkbox("Use Positional Encoding")
    cluster = st.selectbox(
              'Select cluster strategy', ClusterFactory.values_list())
    auto_cluster = st.checkbox("Automatic find the optimal number of clusters")

    cluster_selector = st.empty()
    num_cluster_slider = st.empty()
    video_vis = st.empty()

    timeline_vis = st.empty()
    cluster_vis = st.empty()

    segment_selection = st.empty()
    segment_video = st.empty()

    video_vis.video(video_path)

    if feat_io.has_features is False:
        with st.spinner("Extracting..."):
            progress_bar = st.progress(0)
            for i in feature_extractor.extract(video_path):
                progress_bar.progress(i)
            progress_bar.empty()

    if auto_cluster is False:
        num_cluster_slider = num_cluster_slider.slider(
            "Select number of clusters", min_value=2, max_value=29)

    cluster = ClusterFactory.get(cluster)(n=num_cluster_slider)

    with st.spinner("Clustering..."):
        if pos_enc:
            features = positional_encoding(feat_io.read())
        else:
            features = feat_io.read()

        if auto_cluster:
            c_labels = cluster.auto(features)
        else:
            c_labels = cluster.estimate(features)

    if c_labels is not None:
        df = cluster.df(features, c_labels)
        timeline_chart = plots.video_timeline(
            utils.cluster_to_segment_bounds(df.labels))
        clusters_chart = plots.clusters(df)

        st.altair_chart(alt.vconcat(timeline_chart, clusters_chart))
