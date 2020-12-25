import unittest
from models import SlowFastModel
import data_io


class TestSlowFastModel(unittest.TestCase):

    def setUp(self):
        self.video_path = "./tests/assets/RECartaodeVisitas_2010_ig9-16_14-10_41_Ricardo.mp4"
        # self.video_path = "./tests/assets/RoadAccidents010_x264.mp4"
        self.out_path = "./tests/assets/test_slowfastmodel_get_features"

    def test_slowfastmodel_init(self):
        _ = SlowFastModel()

    def test_slowfastmodel_get_features(self):
        slowfast = SlowFastModel()
        video_io = data_io.VideoIO()
        fps, num_frames = video_io.metadata(self.video_path)
        frames_generator = video_io.reader(self.video_path)

        clip_nframes = 16
        n_clips = int(num_frames // clip_nframes) - 1

        for i in range(n_clips):
            feat_arr = slowfast.get_features(frames_generator, clip_nframes)
            self.assertEqual(feat_arr.shape[0], clip_nframes)


if __name__ == '__main__':
    unittest.main()
