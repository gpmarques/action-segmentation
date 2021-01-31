import altair as alt
import pandas as pd


def video_timeline(segs: list) -> alt.Chart:
    data = pd.DataFrame()
    data["start"] = [list(seg.values())[0][0] for seg in segs]
    data["end"] = [list(seg.values())[0][1] for seg in segs]
    data["label"] = [list(seg.keys())[0] for seg in segs]
    max_end = data.end.max()
    return alt.Chart(data)\
              .mark_bar()\
              .encode(x=alt.X("start",
                              axis=alt.Axis(tickCount=data.end.max()),
                              scale=alt.Scale(domain=(0, max_end))),
                      x2="end",
                      color=alt.Color('label:N',
                                      scale=alt.Scale(scheme='category10'))
                      ).properties(width=680, height=60)


def clusters(df: pd.DataFrame) -> alt.Chart:
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
