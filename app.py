import streamlit as st
import altair as alt
import os

from extractors import FeatureExtraction, SlowFastStrategy
from clusterize import KMeansClusterStrategy
from features import Features
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
    features = Features(video_path)
    video = Video(video_path)
    segmentator = ActionSegmentation(video_path)
    metadata = video.metadata

    cluster = None
    c_labels = None

    show_selected_video = st.sidebar.checkbox("Show selected video")
    if show_selected_video:
        st.video(video_path)

    features_button = st.empty()

    clusterize_button = st.empty()
    num_cluster_slider = st.empty()
    timeline_vis = st.empty()
    cluster_vis = st.empty()

    segment_selection = st.empty()
    segment_video = st.empty()


    if features.has_features is False:
        features_button = features_button.button("Extract features")

        if features_button:
            with st.spinner("Extracting..."):
                progress_bar = st.progress(0)
                for i in feature_extractor.extract(video_path):
                    progress_bar.progress(i)
                progress_bar.empty()
    else:
        num_cluster_slider = num_cluster_slider.slider(
            "Select number of clusters", min_value=2, max_value=12)

        cluster = KMeansClusterStrategy(k=num_cluster_slider,
                                        path=video_path)

        with st.spinner("Clustering..."):
            c_labels, df = cluster.clusterize(features.read())
            # video.clusters_to_videos(c_labels)

        if c_labels is not None:
            timeline_chart = plots.video_timeline(
                utils.cluster_to_segment_bounds(df.cluster))
            # timeline_vis.altair_chart(timeline_chart)
            clusters_chart = plots.clusters(df)
            # cluster_vis.altair_chart(clusters_chart)
            st.altair_chart(alt.vconcat(timeline_chart, clusters_chart))




    # if video.is_segmented:
    #     segment_header.header("Segments from clusters")
    #     cluster = KMeansClusterStrategy(k=2,
    #                                     path=video_path)
    #     df = cluster.get_cluster()
    #     chart = alt.Chart(df).mark_point().encode(alt.X('x:Q'),
    #                                               alt.Y('y:Q'),
    #                                               color=alt.Color("labels:N")
    #                                               ).properties(width=640,
    #                                                            height=480)
    #     cluster_vis.altair_chart(chart)
    #     timeline_vis.altair_chart(
    #         plots.video_timeline(utils.cluster_to_segment_bounds(df.labels)))
    #
    #     segment_selected = segment_selection.selectbox('Select a segment',
    #                                                    video.segments)
    #     segment_video.video(os.path.join(video.segments_dir, segment_selected))
    #
    # elif features.has_features:
    #     num_cluster_slider = st.slider("Select number of clusters",
    #                                    min_value=2,
    #                                    max_value=12)
    #     clusterize_button = clusterize_button.button('Clusterize')
    #     if clusterize_button:
    #         cluster = KMeansClusterStrategy(k=num_cluster_slider,
    #                                         path=video_path)
    #
    #         with st.spinner("Clustering..."):
    #             c_labels, df = cluster.clusterize(features.read())
    #             video.clusters_to_videos(c_labels)
    #
    #     if c_labels is not None:
    #
    #         segment_header.header("Segments from clusters")
    #         segment_selected = segment_selection.selectbox('Select a segment',
    #                                                        video.segments)
    #         segment_video.video(os.path.join(video.segments_dir,
    #                                          segment_selected))
    #
    #         chart = alt.Chart(df).mark_point().encode(
    #             alt.X('x:Q'), alt.Y('y:Q'),
    #             color=alt.Color("labels:N"),
    #         ).properties(width=640, height=480)
    #
    #         cluster_vis.altair_chart(chart)
    #
    #         segs = utils.cluster_to_segment_bounds(df.labels)
    #         timeline_plot = plots.video_timeline(segs)
    #         timeline_vis.altair_chart(timeline_plot)
    #
    #
    # else:
    #     if st.button('Extract features'):
    #         with st.spinner("Extracting..."):
    #             progress_bar = st.progress(0)
    #             for i in feature_extractor.extract(video_path):
    #                 progress_bar.progress(i)
    #             progress_bar.empty()
    #         st.success("Features extracted! Now click on the segment button to segmentate your video.")
    #
    #         if segment_button.button('Segment'):
    #             with st.spinner("Segmenting..."):
    #                 segment = segmentator.segment()
    #
    #             segment_button.empty()
    #             segment_header.header("Segments")
    #             segment_selected = segment_selection.selectbox('Select a segment', video.segments)
    #             segment_video.video(os.path.join(video.segments_dir, segment_selected))
