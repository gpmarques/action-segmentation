import unittest
import numpy as np
from extractors import FeatureExtraction, SlowFastStrategy
from features import Features


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.video_path = "./tests/assets/RoadAccidents010_x264.mp4"
        self.features = Features(self.video_path)
        self.extractor = FeatureExtraction()

    def test_feature_extraction(self):
        self.extractor.extraction_strategy = SlowFastStrategy()
        for i in self.extractor.extract(self.video_path):
            print(i)
        feature_array = self.features.read()
        self.assertIsInstance(feature_array, np.ndarray)


if __name__ == '__main__':
    unittest.main()
