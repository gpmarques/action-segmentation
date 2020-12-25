import altair as alt
import pandas as pd


def video_timeline(segs: list) -> alt.Chart:
    data = pd.DataFrame()
    data["start"] = [list(seg.values())[0][0] / 30.0 for seg in segs]
    data["end"] = [list(seg.values())[0][1] / 30.0 for seg in segs]
    data["cluster"] = [list(seg.keys())[0] for seg in segs]
    max_end = data.end.max()
    return alt.Chart(data)\
              .mark_bar()\
              .encode(x=alt.X("start", scale=alt.Scale(domain=(0, max_end))),
                      x2="end",
                      # y="cluster:N",
                      color=alt.Color('cluster:N',
                                      scale=alt.Scale(scheme='dark2'))
                      ).properties(width=680, height=60)


def clusters(df: pd.DataFrame) -> alt.Chart:
    chart = alt.Chart(df)\
               .mark_point()\
               .encode(alt.X('x:Q'),
                       alt.Y('y:Q'),
                       color=alt.Color("cluster:N",
                                       scale=alt.Scale(scheme='dark2')))\
               .properties(width=680, height=480)
    return chart
