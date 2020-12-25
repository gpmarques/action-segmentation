from data_io import VideoIO
from collections.abc import Generator
import numpy as np
import utils
import os


class Video(object):

    def __init__(self, path: str, BGR=False):
        self._path = path
        self._v_io = VideoIO()
        self._channel_conv = BGR

    @property
    def metadata(self) -> tuple:
        """ Returns a tuple with fps and number of frames """
        return self._v_io.metadata(self._path)

    @property
    def name(self) -> str:
        """ Extract the name of the video file from its path """
        return self._path.split("/")[-1]

    @property
    def segments_dir(self) -> str:
        """ Returns the path of this video's segments directory """
        path = "/".join(self._path.split("/")[:-1])
        seg_folder = "_".join([self.name, "segments"])
        seg_path = os.path.join(path, seg_folder)
        return seg_path

    @property
    def is_segmented(self) -> bool:
        """ Indicater if this video has been segmented or not """
        return self._v_io.count_videos(self.segments_dir)

    @property
    def segments(self) -> list:
        """ Returns a list with all segments of this video """
        segments_fnames = self._v_io.get_videos_filenames(self.segments_dir)
        return sorted(segments_fnames, key=lambda p: int(p.split("_")[1]))

    def frame_generator(self) -> Generator:
        """ Returns a generator of frames """
        return self._v_io.reader(self._path, self._channel_conv)

    def clusters_to_videos(self, clusters: np.ndarray) -> list:
        segs = utils.cluster_to_segment_bounds(clusters)
        frame_gen = self.frame_generator()
        seg_paths = []

        for seg in segs:
            seg_frames = []
            cluster, bounds = list(seg.items())[0]
            start, end = bounds
            while True:
                frame_counter, frame = next(frame_gen)
                if frame_counter >= start and frame_counter < end:
                    seg_frames.append(frame)
                if frame_counter == end:
                    seg_frames.append(frame)
                    break
            seg_path = self.write_segment(
                np.array(seg_frames), start, end, cluster)
            seg_paths.append(seg_path)
        seg_paths.sort(key=lambda p: int(p.split("/")[-1].split("_")[1]))
        return seg_paths

    def write_segment(self, frames: np.ndarray,
                      start: int, end: int, cluster: int) -> str:
        fname = "_".join([str(cluster), str(start), str(end), self.name])

        if os.path.isdir(self.segments_dir) is False:
            os.mkdir(self.segments_dir)

        seg_path = os.path.join(self.segments_dir, fname)
        self._v_io.write(frames, seg_path)
        return seg_path
