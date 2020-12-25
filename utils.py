import numpy as np


def cluster_to_segment_bounds(clusters: np.ndarray):
    """  Map clustered video segments into continuos segments of each cluster

    args:
    data - label of each segment
    labels - the unique labels

    return:
    segs - list of dicts, where each key is a label and a tuple is the start
           and end of a continuous segment
    """

    segs = []
    for label in np.unique(clusters):
        idxs = np.where(clusters == label)[0]
        curr = idxs[0]
        last = None
        step = 0
        for idx in idxs:
            if idx != curr + step:
                segs.append({label: (curr, last)})
                curr = idx
                step = 1
                last = None
            else:
                last = idx
                step += 1

        segs.append({label: (curr, idx)})
    segs.sort(key=lambda value: list(value.values())[0])
    return segs
