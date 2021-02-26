import numpy as np

import cv2 as opencv

from Types import FeatureExtractorType


class VisualFeatureExtrator:
    def __init__(self, type_xfe: FeatureExtractorType = FeatureExtractorType.SURF, verbose=False):
        self.type = type_xfe
        self.extractor = None

        self.__verbose = verbose
        self.__initialized = False

    def __del__(self):
        self.reset()

    def reset(self):
        del self.extractor
        self.__initialized = False

    @property
    def type(self) -> FeatureExtractorType:
        return self.__type_xfe

    @type.setter
    def type(self, type_xfe: FeatureExtractorType):
        self.__type_xfe = type_xfe

    @property
    def initialized(self) -> bool:
        return self.__initialized

    def __initExtractor(self):
        if self.type == FeatureExtractorType.SURF or FeatureExtractorType.DEFAULT:
            self.extractor = opencv.xfeatures2d.SURF_create(100, 4, 3, 0, 0)
        elif self.type == FeatureExtractorType.SIFT:
            self.extractor = opencv.xfeatures2d.SIFT_create(0, 3, 0.03, 10, 1.6)
        else:
            raise Exception('Not supported type of feature extractor')

        self.__initialized = True

    def fit(self, img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        if not self.__initialized:
            self.__initExtractor()

        __, descriptors = self.extractor.detectAndCompute(img, mask)
        # TODO check descriptors type
        return descriptors



