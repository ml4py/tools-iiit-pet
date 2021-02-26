from enum import Enum, auto

import cv2 as opencv
import numpy as np


class DatasetType(Enum):
    TRAINING = 'trainval'
    TEST = 'test'


class PetFamily(Enum):
    NONE = 0
    CAT = 1
    DOG = -1


class ROI(Enum):
    ALL = auto()
    PET = auto()
    FACE = auto()


class OPENCV_NORM(Enum):
    HAMMING = opencv.NORM_HAMMING
    HAMMING2 = opencv.NORM_HAMMING2
    L1 = opencv.NORM_L1
    L2 = opencv.NORM_L2
    L2SQR = opencv.NORM_L2SQR
    INF = opencv.NORM_INF
    MINMAX = opencv.NORM_MINMAX


class OPENCV_IMREAD(Enum):
    GRAYSCALE = opencv.IMREAD_GRAYSCALE
    COLOR = opencv.IMREAD_COLOR


class FORMAT(Enum):
    HDF5 = 'hdf5'
    JPG = 'jpg'
    JPEG = 'jpeg'
    PNG = 'png'
    TIFF = 'tiff'


class FeatureExtractorType(Enum):
    SIFT = auto()
    SURF = auto()
    ORB = auto()
    DEFAULT = auto()


class ImgTransformation(Enum):
    NONE = -1
    SCALE_MEAN = 1
    SCALE_MEDIAN = 2
    SCALE_MIN = 3
    SCALE_MAX = 4
    FIT_MAX = 5


class FeaturesType(Enum):
    SPARSE = auto()
    DENSE = auto()


def imgIsColored(img) -> bool:
    return False if len(img.shape) == 2 else True


def vstackNumpyArrays(input):
    out = input[0]
    for i in range(1, len(input)):
        if input[i] is not None and input[i].shape[0] > 0:
            out = np.vstack((out, input[i]))
    return out
