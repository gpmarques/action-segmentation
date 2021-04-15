import unittest
from video import Video
import math
import numpy as np


class TestVideo(unittest.TestCase):

    def setUp(self):
        self.video_path = "./tests/assets/video/RoadAccidents010_x264.mp4"
        self.video = Video(self.video_path, "SlowFast")

    def test_video_len(self):
        assert math.floor(len(self.video) / 30) == 17

    def test_video_call(self):
        idx = range(0, 30, 2)
        frames = self.video(idx)
        assert isinstance(frames, np.ndarray)
        assert frames.shape == (15, 240, 320, 3)

    def test_video_fps(self):
        assert self.video.fps == 30.0

    def test_video_duration(self):
        assert self.video.duration == 17.6


if __name__ == '__main__':
    unittest.main()
