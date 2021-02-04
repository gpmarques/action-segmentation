""" Npy module

This module implements all npy file management class and methods

...
Classes
-------
Npy
    Class that manages npy file's reading and writing functionalities
"""

import numpy as np
import os


class Npy(object):
    """ Class that manages npy file's reading and writing functionalities

    Methods
    -------
    read(path: str)
        Method that reads all npy files in the dir's path. Npy files in this dir
        should be form one numpy array when read, this method is going to create
        one np array with the npy files found

    write(path: str, npy_fname: str, npy: np.ndarray)
        Method that write a new npy file in the path passed with the npy_fname as
        the file's name

    count_npy_files(path: str)
        Method that counts the number of npy files in a dir
    """

    def read(self, path: str) -> np.ndarray:
        """ Method to read all npy files in the path passed. It expects npy fi-
        -les to be in the following format XXXX_<name>.npy, where XXXX is a
        number ordering the files

        Parameters
        ----------
        path: str
            path to the folder containing the npy files

        Returns
        --------
        npy_arrays: np.ndarray
            np.array with shape (N, S) where N is the number of files in
            the path and S is the shape of each npy file read
        """
        npy_fnames = os.listdir(path)  # getting all npy files in path

        assert len(npy_fnames) != 0, "No files in path {0}".format(path)

        npy_fnames = [  # making sure each file is npy with index prefix
            fn for fn in npy_fnames if self.__assert_is_npy(fn)
        ]

        assert len(npy_fnames) == len(os.listdir(path)), "There must be only npy files in path {0}".format(path)

        # sorting the files by the index prefix in each file name
        npy_fnames = sorted(npy_fnames, key=lambda p: int(p.split("_")[0]))
        features = [np.load(os.path.join(path, fname))
                    for fname in npy_fnames]
        features = np.array(features)
        return features.reshape((features.shape[0] * features.shape[1],
                                 features.shape[2]))

    def write(self, path: str, npy_fname: str, npy: np.ndarray) -> None:
        """ Method to write a npy file in the path passed

        Parameters:
        path: str
            path to the folder to write the new npy file
        npy_fname: str
            name of the new npy file
        npy: np.ndarray
            content of the new npy file
        """
        self.__assert_is_npy(npy_fname)

        if os.path.isdir(path) is False:
            os.mkdir(path)

        write_path = os.path.join(path, npy_fname)
        np.save(write_path, npy)

    def count_npy_files(self, path: str) -> int:
        """ Method that counts the number of files in a folder

        Parameters
        ----------
        path: str
            path to the folder where the method should count the number of npy
            files

        Returns
        -------
        int
            Returns the number of files in a folder
        """
        if os.path.isdir(path) is False:
            return 0

        return len([
            fn for fn in os.listdir(path) if self.__assert_is_npy(fn)
        ])

    def __assert_is_npy(self, fname: str):
        """ Method that returns true if the extension of a file name is npy

        Parameters
        ----------
        fname: str
            a file name

        Returns
        -------
        boolean
            Returns True if the extension is npy otherwise False
        """
        return "npy" == fname.split(".")[-1]
