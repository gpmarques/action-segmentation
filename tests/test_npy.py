import unittest
from npy import Npy
import numpy as np
import os


class TestNpyIO(unittest.TestCase):

    def setUp(self):
        self.npy_path = "./tests/assets/test_npy/read"
        self.out_path = "./tests/assets/test_npy/write"
        self.fname = "001_test.npy"
        self.npy_test = np.array([[1, 1, 1]])
        np.save(os.path.join(self.npy_path, self.fname), self.npy_test)

    def test_npy_read(self):
        npy_io = Npy()
        npy = npy_io.read(self.npy_path)
        self.assertTrue(self.npy_test.shape == npy.shape)

    def test_npy_write_n_count(self):
        npy_io = Npy()
        npy_io.write(self.out_path, self.fname, self.npy_test)
        npy = npy_io.read(self.out_path)
        assert npy_io.count_npy_files(self.out_path) == 1
        assert self.npy_test.shape == npy.shape

    def tearDown(self):
        if os.path.exists(os.path.join(self.out_path, self.fname)):
            os.remove(os.path.join(self.out_path, self.fname))


if __name__ == '__main__':
    unittest.main()
