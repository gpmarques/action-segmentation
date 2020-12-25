import numpy as np
from imutils.video import FileVideoStream
import cv2
import os
from typing import Iterator
import subprocess


class Npy(object):
    """ This module manages npy file's reading and writing functionalities """

    def read(self, path: str) -> np.ndarray:
        """ Method to read all npy files in the path passed. It expects npy fi-
        -les to be in the following format XXXX_<name>.npy, where XXXX is a
        number ordering the files

        Args:
            path: path to the folder containing the npy files
        Returns:
            npy_arrays: np.array with shape (N, S) where N is the
                        number of files in the path and S is the
                        shape of each npy file read
        """
        npy_fnames = os.listdir(path)  # getting all npy files in path

        assert len(npy_fnames) != 0, "No files in path {0}".format(path)

        npy_fnames = [  # making sure each file is npy with index prefix
            fn for fn in npy_fnames if self.__assert_is_npy(fn)
        ]

        assert len(npy_fnames) == len(os.listdir(path)), "There must be only npy files in path {0}".format(path)

        # sorting the files by the index prefix in each file name
        npy_fnames = sorted(npy_fnames, key=lambda p: int(p.split("_")[0]))
        npy_arrays = np.array(
            [np.load(os.path.join(path, fname))
             for fname in npy_fnames]
        )

        return npy_arrays

    def write(self, path: str, npy_fname: str, npy: np.ndarray) -> None:
        """ Method to write a npy file in the path passed

        Args:
            path: path to the folder to write the new npy file
            npy_fname: name of the new npy file
            npy: content of the new npy file
        """
        self.__assert_is_npy(npy_fname)

        if os.path.isdir(path) is False:
            os.mkdir(path)

        write_path = os.path.join(path, npy_fname)
        np.save(write_path, npy)

    def count_npy_files(self, path: str) -> int:
        if os.path.isdir(path) is False:
            return 0

        return len([
            fn for fn in os.listdir(path) if self.__assert_is_npy(fn)
        ])

    def __assert_is_npy(self, fname: str):
        return "npy" == fname.split(".")[-1]


class VideoIO:
    """ This module manages video's reading and writing functionalities """

    def __init__(self):
        self.fvs = None

    def reader(self, path: str, BGR=False) -> Iterator:
        """ Generator function that reads a video frame by frame, yielding a
        tuple containing the frame index and the frame itself

        Args:
        path: the path to the video
        BGR: flag to signal if mantains the channel convention BGR
        Returns:
        A generator that yields a tuple with an integer and a numpy array
        """
        self.fvs = FileVideoStream(path).start()
        ith_frame = 0
        while self.fvs.more():
            frame = self.fvs.read()
            if frame is not None:
                # If BGR is set to False convert the frame to RGB convention
                yield (ith_frame, frame if BGR else frame[:, :, ::-1])
            else:
                break
            ith_frame += 1
        self.fvs.stop()

    def write(self, frames: np.ndarray, path: str) -> None:
        """Write a numpy array of frames into a mp4 video

        Args:
        frames: numpy array with shape (number of frames, height, width,
                  channels)
        path: the path you want to write the video to
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]

        splitted_path = path.split("/")
        path_to_video, filename = splitted_path[:-1], splitted_path[-1]
        temp_path = os.path.join(*path_to_video, "_".join(["temp", filename]))

        out = cv2.VideoWriter(
            temp_path, fourcc, 30.0, (width, height))
        for frame in frames:
            out.write(frame[:, :, ::-1])
        out.release()

        self.convert_codec(temp_path, path)
        os.remove(temp_path)

    def metadata(self, path: str) -> tuple:
        """ Returns fps and number of frames of a video in the given path """
        video = cv2.VideoCapture(path)
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver) < 3:  # if opencv version is less than 3
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
            total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        else:
            fps = video.get(cv2.CAP_PROP_FPS)
            total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return fps, total

    def get_videos_filenames(self, path: str) -> list:
        if os.path.isdir(path) is False:
            return []

        filenames = os.listdir(path)
        filenames = [
            filename for filename in filenames
            if filename.split(".")[-1] in ["mp4"]]

        return filenames

    def count_videos(self, path: str) -> int:
        return len(self.get_videos_filenames(path))

    def convert_codec(
            self, in_path: str, out_path: str,
            new_codec: str = "libx264") -> None:
        command = ["ffmpeg", "-i", in_path, "-vcodec", new_codec, out_path]
        process = subprocess.Popen(command)
        process.wait()

    def play(self, path: str) -> None:
        """ Plays a video """
        for frame in self.read(path):
            if frame is None:
                break
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
