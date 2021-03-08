import numpy as np

import cv2 as opencv

from Types import FeatureExtractorType


class VisualFeatureExtractor:
    def __init__(self, type_xfe: FeatureExtractorType = FeatureExtractorType.SURF, verbose=False):
        self.type = type_xfe
        self.extractor = None

        self.__verbose = verbose
        self.__initialized = False

        self.SIFT_nfeatures = 0
        self.SIFT_nOctaveLayers = 3
        self.SIFT_contrastThreshold = 0.4
        self.SIFT_edgeThreshold = 10
        self.SIFT_sigma = 1.6

        self.SURF_hessianThreshold = 100
        self.SURF_nOctaves = 4
        self.SURF_nOctaveLayers = 3
        self.SURF_extended = False
        self.SURF_upright = False

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

    #SIFT Parameters
    @property
    def SIFT_nfeatures(self) -> int:
        return self.__SIFT_nfeatures

    @SIFT_nfeatures.setter
    def SIFT_nfeatures(self, nfeatures: int):
        self.__SIFT_nfeatures = nfeatures

    @property
    def SIFT_nOctaveLayers(self) -> int:
        return self.__SIFT_nOctaveLayers

    @SIFT_nOctaveLayers.setter
    def SIFT_nOctaveLayers(self, nOctaveLayers: int):
        self.__SIFT_nOctaveLayers = nOctaveLayers

    @property
    def SIFT_contrastThreshold(self) -> float:
        return self.__SIFT_contrastThreshold

    @SIFT_contrastThreshold.setter
    def SIFT_contrastThreshold(self, contrastThreshold: float):
        self.__SIFT_contrastThreshold = contrastThreshold

    @property
    def SIFT_edgeThreshold(self) -> int:
        return self.__SIFT_edgeThreshold

    @SIFT_edgeThreshold.setter
    def SIFT_edgeThreshold(self, edgeThreshold: int):
        self.__SIFT_edgeThreshold = edgeThreshold

    @property
    def SIFT_sigma(self) -> float:
        return self.__SIFT_sigma

    @SIFT_sigma.setter
    def SIFT_sigma(self, sigma: float):
        self.__SIFT_sigma = sigma

    #SURF Parameters
    @property
    def SURF_hessianThreshold(self) -> float:
        return self.__SURF_hessianThreshold

    @SURF_hessianThreshold.setter
    def SURF_hessianThreshold(self, hessianThreshold: float):
        self.__SURF_hessianThreshold = hessianThreshold

    @property
    def SURF_nOctaves(self) -> int:
        return self.__SURF_nOctaves

    @SURF_nOctaves.setter
    def SURF_nOctaves(self, nOctaves: int):
        self.__SURF_nOctaves = nOctaves

    @property
    def SURF_nOctaveLayers(self) -> int:
        return self.__SURF_nOctaveLayers

    @SURF_nOctaveLayers.setter
    def SURF_nOctaveLayers(self, nOctaveLayers: int):
        self.__SURF_nOctaveLayers = nOctaveLayers

    @property
    def SURF_extended(self) -> bool:
        return self.__SURF_extended

    @SURF_extended.setter
    def SURF_extended(self, extended: bool):
        self.__SURF_extended = extended

    @property
    def SURF_upright(self) -> bool:
        return self.__SURF_upright

    @SURF_upright.setter
    def SURF_upright(self, upright: bool):
        self.__SURF_upright = upright

    @property
    def initialized(self) -> bool:
        return self.__initialized

    def __initExtractor(self):
        if self.type == FeatureExtractorType.SURF or FeatureExtractorType.DEFAULT:
            self.extractor = opencv.xfeatures2d.SURF_create(self.SURF_hessianThreshold, self.SURF_nOctaves, self.SURF_nOctaveLayers, self.SURF_extended, self.SURF_upright)
        elif self.type == FeatureExtractorType.SIFT:
            self.extractor = opencv.xfeatures2d.SIFT_create(self.SIFT_nfeatures, self.SIFT_nOctaveLayers, self.SIFT_contrastThreshold, self.SIFT_edgeThreshold, self.SIFT_sigma)
        else:
            raise Exception('Not supported type of feature extractor')

        self.__initialized = True

    def fit(self, img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        if not self.__initialized:
            self.__initExtractor()

        __, descriptors = self.extractor.detectAndCompute(img, mask)
        # TODO check descriptors type
        return descriptors



