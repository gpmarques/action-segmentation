import unittest
from segment_video import ActionSegmentation
import os
from video import Video


class TestActionSegmentation(unittest.TestCase):

    def setUp(self):
        self.feat_path =\
            "./tests/assets/test_feat_extractors/TEST"
        self.video_path = "./tests/assets/RoadAccidents010_x264.mp4"
        self.segments_paths = "./tests/assets/segments"

    def test_compute_change_points(self):
        action_segmentation = ActionSegmentation(
            self.video_path, self.feat_path)
        action_segmentation.segment()
        segments_paths = os.listdir(self.segments_paths)
        for seg_path in segments_paths:
            print(seg_path)
            start, end = seg_path.split("_")[:2]
            number_of_frames = int(end) - int(start) + 1
            full_seg_path = os.path.join(self.segments_paths, seg_path)
            v = Video(full_seg_path)
            self.assertEqual(v.metadata[1], number_of_frames)
            os.remove(full_seg_path)

    def tearDown(self):
        os.rmdir(self.segments_paths)


if __name__ == '__main__':
    unittest.main()
