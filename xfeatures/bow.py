from typing import Union

import numpy as np
from scipy.spatial import cKDTree

import cv2 as opencv


class VisualDictionary:
    def __init__(self, features: np.ndarray = None, verbose=False):
        self.training_features = features
        self.codebook_size = 64  # TODO default

        self.__codebook = None
        self.__codebook_tree = None

        self.__verbose = verbose
        self.__traincalled = False

    def __del__(self):
        del self.training_features
        del self.__codebook
        del self.__codebook_tree

    @property
    def training_features(self) -> np.ndarray:
        return self.__img_features

    @training_features.setter
    def training_features(self, features: np.ndarray):
        self.__img_features = features

    @training_features.deleter
    def training_features(self):
        del self.__img_features

    @property
    def codebook_size(self) -> int:
        return self.__codebook_size

    @codebook_size.setter
    def codebook_size(self, size: int):
        self.__codebook_size = size

    def train(self):
        if self.__traincalled:
            return

        del self.__codebook, self.__codebook_tree

        # option for choosing vector quantification technique
        criteria = (opencv.TERM_CRITERIA_EPS + opencv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        rss, _, self.__codebook = opencv.kmeans(self.training_features, self.codebook_size, None, criteria, 5,
                                                opencv.KMEANS_PP_CENTERS)

        self.__codebook_tree = cKDTree(self.__codebook)
        self.__traincalled = True

    def generateCodeword(self, features: np.ndarray) -> np.ndarray:
        if not self.__traincalled:
            self.train()

        _, idx = self.__codebook_tree.query(features)
        codeword, _ = np.histogram(idx, bins=self.codebook_size + 1, range=(0, self.codebook_size))

        return codeword

    def fit(self, features: Union[np.ndarray, list]) -> Union[np.ndarray, list]:
        if type(features) == np.ndarray:
            return self.generateCodeword(features)
        else:
            codewords = []
            for f in features:
                codeword = self.generateCodeword(f)
                codewords.append(codeword)

            return codewords

# TODO setFromOptions
