""" Video segmentation module

This module reads the npy files with the features from a video and segments the
video

"""
import numpy as np
from external.KTS.cpd_auto import cpd_auto
from video import Video
from features import Features


class ActionSegmentation(object):

    def __init__(self, video_path: str, max_cps=6):
        self._video = Video(video_path)
        self._features = Features(video_path)
        self._max_cps = max_cps

    def compute_change_points(self, features: np.ndarray) -> tuple:
        """ Given a 2D numpy array of features this method computes the change
        points for a video with the algorithm KTS
        Args:
            features: 2D numpy array representing the features of a video
        Returns:
            a tuple with a list of the change points and a list with scores of
            each change point. The change points are the x indices in the fea-
            -tures numpy array.
        """
        cps, scores = cpd_auto(
            np.dot(features, features.T), self._max_cps, vmax=1)
        return cps, scores

    def _cps_to_segments_boundaries(self,
                                    cps: list,
                                    number_of_features: int) -> list:
        segments = [(0, cps[0])]
        for i, cp in enumerate(cps[1:]):
            segments.append((segments[-1][1] + 1, cp))
        segments.append((cp + 1, number_of_features))
        return segments

    def segment(self) -> None:
        features = self._features.read()

        cps, _ = self.compute_change_points(features)
        segments_bounds = self._cps_to_segments_boundaries(
            cps, features.shape[0])

        frame_gen = self._video.frame_generator()
        segments_paths = []
        for start, end in segments_bounds:
            seg_frames = []
            while True:
                frame_counter, frame = next(frame_gen)
                if frame_counter >= start and frame_counter < end:
                    seg_frames.append(frame)
                if frame_counter == end:
                    seg_frames.append(frame)
                    break
            seg_path = self._video.write_segment(
                np.array(seg_frames), start, end)
            segments_paths.append(seg_path)
        return segments_paths
