""" Video Module

This module contains the class that represents a video

Classes
-------

Video
    Class that represents a video
"""
import decord
import cv2
from features import Features
import numpy as np


class Video(object):
    """ Class that represents a video

    Attributes
    ----------
    _path: str
        Path to this video

    reader: decord.VideoReader
        Object that reads this video's frames

    features: Features
        Object that manages the features extracted from this video

    name: str
        Filename of this video

    Methods
    -------
    __len__()
        Returns the total number of frames this video has

    __call__(frame_id_list: list)
        Returns the frames from the parameter frame_id_list
    """

    def __init__(self, path: str, extractor: str):
        """
        Parameters
        ----------
        path: str
            The path to this video

        extractor: str
            The name of the feature extractor currently used
        """
        self._path = path
        self.reader = decord.VideoReader(self._path)
        self.features = Features(self._path, extractor)

    @property
    def name(self) -> str:
        """ Returns the name of the video file from its path """
        return self._path.split("/")[-1]

    @property
    def fps(self) -> float:
        video = cv2.VideoCapture(self._path)

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver) < 3:
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps

    @property
    def duration(self) -> float:
        return len(self) / self.fps

    def __len__(self) -> int:
        """ Returns the number of frames of this video """
        return len(self.reader)

    def __call__(self, frame_id_list: list) -> np.ndarray:
        """ Returns the frames based on the indexes passed in the frame_id_list

        Parameters
        ----------
        frame_id_list: list
            List containing indexes of frames from this video

        Returns
        -------
        np.ndarray
            Returns a numpy array containing the frames from the index of this
            method parameter
        """
        return self.reader.get_batch(frame_id_list).asnumpy()
