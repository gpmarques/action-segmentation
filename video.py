import decord
from features import Features
import numpy as np


class Video(object):

    def __init__(self, path: str, extractor: str):
        self._path = path
        self.reader = decord.VideoReader(self._path)
        self.features = Features(self._path, extractor)

    @property
    def name(self) -> str:
        """ Returns the name of the video file from its path """
        return self._path.split("/")[-1]

    def __len__(self) -> int:
        """ Returns the number of frames of this video """
        return len(self.reader)

    def __call__(self, frame_id_list: list) -> np.ndarray:
        """ Returns the frames based on the indexes passed in the frame_id_list
        Args:
        frame_id_list - list containing indexes of frames from this video

        Returns:

        """
        return self.reader.get_batch(frame_id_list).asnumpy()
