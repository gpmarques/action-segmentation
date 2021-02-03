import unittest
import shutil
import os
import numpy as np
from extractors import SlowFastStrategy, I3DStrategy
from extractors import ExtractorFactory
from video import Video


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.video_path = "./tests/assets/RoadAccidents010_x264.mp4"

    def test_slowfast(self):
        slowfast = SlowFastStrategy()
        video = Video(self.video_path, ExtractorFactory.SLOWFAST.value)

        slowfast.extract(video)
        features = video.features()

        self.assertIsInstance(features, np.ndarray)
        assert features.shape == (17, 2304)

    def test_i3d(self):
        i3d = I3DStrategy()
        video = Video(self.video_path, ExtractorFactory.I3D.value)

        i3d.extract(video)
        features = video.features()

        self.assertIsInstance(features, np.ndarray)
        assert features.shape == (17, 2048)

    def test_cluster_factory(self):
        extractors = ExtractorFactory.values_list()
        assert isinstance(ExtractorFactory.get(extractors[0])(),
                          SlowFastStrategy)
        assert isinstance(ExtractorFactory.get(extractors[1])(),
                          I3DStrategy)

    def tearDown(self):
        if os.path.isdir(self.video_path+"_features"):
            shutil.rmtree(self.video_path+"_features")


if __name__ == '__main__':
    unittest.main()
