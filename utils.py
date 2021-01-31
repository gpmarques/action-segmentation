import numpy as np


def cluster_to_segment_bounds(clusters: np.ndarray):
    """  Map clustered video segments into continuos segments of each cluster

    args:
    clusters - labels of each segment

    return:
    segs - list of dicts, where each key is a label and a tuple is the start
           and end of a continuous segment
    """

    segs = []
    for label in np.unique(clusters):
        idxs = np.where(clusters == label)[0]

        first_idx = idxs[0]
        for i_idx, idx in enumerate(idxs):
            if i_idx == 0:
                continue

            if idx != idxs[i_idx] + 1:
                segs.append({label: (first_idx, idx + 1)})
                first_idx = idx

        segs.append({label: (first_idx, idx + 1)})
    segs.sort(key=lambda value: list(value.values())[0])
    print(segs)
    return segs
