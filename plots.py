"""
Module containing methods that creates charts

Methods
-------
video_timeline(segs: list)
    Method that creates an altair bar chart representing a video timeline, where
    each continuous segment listed in the parameter sigs is colored based on the
    cluster it belongs

clusters(df: pd.DataFrame)
    Method that creates an altair scatter chart with each feature colored based
    on the cluster it belongs
"""
import altair as alt
import pandas as pd


def video_timeline(segs: list) -> alt.Chart:
    """
    Method that creates an altair bar chart representing a video timeline, where
    each continuous segment listed in the parameter sigs is colored based on the
    cluster it belongs. Segs is a list of dicts where each dict follows this
    pattern {cluster_label: (start_of_segment, end_of_segment)}, therefore sigs
    stores all continuous segments for each cluster label.

    Parameters
    ----------
    segs: list
        list of dicts containing all continuous segments of each cluster label

    Returns
    -------
    alt.Chart
        Bar chart resembling a video timeline where all segments are colored
        based on the cluster it belongs
    """
    data = pd.DataFrame()
    data["start"] = [list(seg.values())[0][0] for seg in segs]
    data["end"] = [list(seg.values())[0][1] for seg in segs]
    data["label"] = [list(seg.keys())[0] for seg in segs]
    max_end = data.end.max()
    return alt.Chart(data)\
              .mark_bar()\
              .encode(x=alt.X("start",
                              axis=alt.Axis(tickCount=max_end),
                              scale=alt.Scale(domain=(0, max_end), clamp=True)),
                      x2="end",
                      color=alt.Color('label:N',
                                      scale=alt.Scale(scheme='category10'))
                      ).properties(width=680, height=60)


def clusters(df: pd.DataFrame) -> alt.Chart:
    """
    Method that creates an altair scatter chart with each feature colored based
    on the cluster it belongs

    Parameters
    ----------
    df: DataFrame
        Dataframe with three columns x, y and labels, where each labels stores
        the colors of each sample in this dataframe

    Returns
    -------
    alt.Chart
        Scatter chart with the df's data
    """
    chart = alt.Chart(df)\
               .mark_point()\
               .encode(alt.X('x:Q'),
                       alt.Y('y:Q'),
                       color=alt.Color("labels:N",
                                       scale=alt.Scale(scheme='category10')))\
                .transform_calculate(
                jitter='sqrt(-2*log(random()))*cos(2*PI*random())')\
               .properties(width=680, height=480)

    return chart
