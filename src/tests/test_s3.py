import unittest
from s3 import S3
import os


class TestS3(unittest.TestCase):

    def setUp(self):
        self.s3 = S3()

    def test_list_objects(self):
        objs = self.s3.list_objects("segmentation-videos", 'videos', ['mp4'])
        assert len(objs) == 1

    def test_download_object(self):
        final_path = "./tests/assets/RoadAccidents010_x264.mp4"
        self.s3.download_object("segmentation-videos",
                                "videos/RoadAccidents010_x264.mp4",
                                "./tests/assets")
        assert os.path.exists(final_path)
        os.remove(final_path)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
