import unittest
from dataio.video import VideoIO
import numpy as np
import os


class TestVideoIO(unittest.TestCase):

    def setUp(self):
        self.video_path = "./tests/assets/RoadAccidents010_x264.mp4"
        self.out_path = "./tests/assets/test_videoio/write/001_segment.mp4"

    def test_v_read(self):
        v_io = VideoIO()
        fps, frame_count = v_io.metadata(self.video_path)
        frame_gen = v_io.reader(self.video_path)
        i, fst_frame = next(frame_gen)
        self.assertIsInstance(fst_frame, np.ndarray)
        for i, _ in frame_gen:
            continue
        self.assertTrue(i+1 == frame_count)

    def test_v_write(self):
        v_io = VideoIO()
        frame_gen = v_io.reader(self.video_path)
        frames = []
        for i, frame in frame_gen:
            if i == 31:
                break
            frames.append(frame)
        frames = np.array(frames)
        v_io.write(frames, self.out_path)
        fps, num_frames = v_io.metadata(self.out_path)
        self.assertEqual(31, num_frames)
        os.remove(self.out_path)


if __name__ == '__main__':
    unittest.main()
