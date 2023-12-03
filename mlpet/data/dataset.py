import csv
import glob
import os
import pathlib

import mlpet.utils.sys as mlsys

from typing import Union

from mlpet.data.pet import ObjPet, PetFamily
from mlpet.io.hdf5 import ViewerHDF5

from mlpet.features.extractor import VisualFeatureExtractor
from mlpet.features.bow import VisualDictionary

# import utils
from mlpet.utils.functools import lazy_import
from mlpet.utils.types import DatasetType, FORMAT, ImgTransformation, imgIsColored, OPENCV_NORM, OPENCV_IMREAD, ROI
from mlpet.utils.types import FeatureExtractorType, FeaturesType, vstackNumpyArrays

# lazy imports
opencv = lazy_import('cv2')
np = lazy_import('numpy')


class Dataset(object):

    def __init__(self, dir_img=None, dir_trimap=None, dir_xml=None, file_training=None, file_test=None, verbose=False):

        # training dataset
        self.__cats_training = []
        self.__dogs_training = []

        # test dataset
        self.__cats_test = []
        self.__dogs_test = []

        self.dir_img = dir_img
        self.dir_trimap = dir_trimap
        self.dir_xml = dir_xml

        self.file_training = file_training
        self.file_test = file_test

        self.output_dir = None

        self.img_roi = ROI.PET
        self.extract_foreground = True
        self.fg_include_border = False
        self.mode_imread = OPENCV_IMREAD.GRAYSCALE
        self.img_transformation = ImgTransformation.NONE
        self.img_centered = False
        self.img_shape = None

        self.extract_features = False
        self.xfeature_type = FeatureExtractorType.DEFAULT
        self.__visual_dictionary = None
        self.__codebook_size = 64  # TODO default value
        self.__codebook_normalize = False
        self.__cbtrained = False

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

        self.ORB_nfeatures = 500
        self.ORB_scaleFactor = 1.2
        self.ORB_nlevels = 8
        self.ORB_edgeThreshold = 31
        self.ORB_firstLevel = 0
        self.ORB_WTA_K = 2
        self.ORB_scoreType = opencv.ORB_HARRIS_SCORE
        self.ORB_patchSize = 31
        self.ORB_fastThreshold = 20

        self.verbose = verbose

    def __del__(self):

        del self.__cats_training
        del self.__dogs_training

        del self.__cats_test
        del self.__dogs_test

        del self.__visual_dictionary

        self.__cbtrained = False

    @property
    def extract_features(self) -> bool:

        return self.__extract_features

    @extract_features.setter
    def extract_features(self, f: bool):

        self.__extract_features = f

    # TODO change name
    @property
    def xfeature_type(self) -> FeatureExtractorType:

        return self.__feature_extractor_type

    @xfeature_type.setter
    def xfeature_type(self, xtype: FeatureExtractorType):

        self.__feature_extractor_type = xtype

    """
    SIFT Parameters
    """

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

    """
    SURF Parameters
    """

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

    """
    ORB Parameters
    """

    @property
    def ORB_nfeatures(self) -> int:

        return self.__ORB_nfeatures

    @ORB_nfeatures.setter
    def ORB_nfeatures(self, nfeatures: int):

        self.__ORB_nfeatures = nfeatures

    @property
    def ORB_scaleFactor(self) -> float:

        return self.__ORB_scaleFactor

    @ORB_scaleFactor.setter
    def ORB_scaleFactor(self, scaleFactor: float):

        self.__ORB_scaleFactor = scaleFactor

    @property
    def ORB_nlevels(self) -> int:

        return self.__ORB_nlevels

    @ORB_nlevels.setter
    def ORB_nlevels(self, nlevels: int):

        self.__ORB_nlevels = nlevels

    @property
    def ORB_edgeThreshold(self) -> int:

        return self.__ORB_edgeThreshold

    @ORB_edgeThreshold.setter
    def ORB_edgeThreshold(self, edgeThreshold: int):

        self.__ORB_edgeThreshold = edgeThreshold

    @property
    def ORB_firstLevel(self) -> int:

        return self.__ORB_firstLevel

    @ORB_firstLevel.setter
    def ORB_firstLevel(self, firstLevel: int):

        self.__ORB_firstLevel = firstLevel

    @property
    def ORB_WTA_K(self) -> int:

        return self.__ORB_WTA_K

    @ORB_WTA_K.setter
    def ORB_WTA_K(self, WTA_K: int):

        self.__ORB_WTA_K = WTA_K

    @property
    def ORB_scoreType(self) -> int:

        return self.__ORB_scoreType

    @ORB_scoreType.setter
    def ORB_scoreType(self, scoreType: int):

        self.__ORB_scoreType = scoreType

    @property
    def ORB_patchSize(self) -> int:

        return self.__ORB_patchSize

    @ORB_patchSize.setter
    def ORB_patchSize(self, patchSize: int):

        self.__ORB_patchSize = patchSize

    @property
    def ORB_fastThreshold(self) -> int:

        return self.__ORB_fastThreshold

    @ORB_fastThreshold.setter
    def ORB_fastThreshold(self, fastThreshold: int):

        self.__ORB_fastThreshold = fastThreshold

    @property
    def xfeature_codebook_size(self) -> int:

        return self.__codebook_size

    @xfeature_codebook_size.setter
    def xfeature_codebook_size(self, size: int):

        self.__codebook_size = size

    @property
    def xfeature_codebook_normalize(self) -> bool:

        return self.__codebook_normalize

    @xfeature_codebook_normalize.setter
    def xfeature_codebook_normalize(self, normalize: bool):

        self.__codebook_normalize = normalize

    @property
    def img_roi(self) -> ROI:

        return self.__img_roi

    @img_roi.setter
    def img_roi(self, roi):

        self.__img_roi = roi

    @property
    def extract_foreground(self) -> bool:

        return self.__extract_foreground

    @extract_foreground.setter
    def extract_foreground(self, f: bool):

        self.__extract_foreground = f

    @property
    def fg_include_border(self) -> bool:

        return self.__fg_border

    @fg_include_border.setter
    def fg_include_border(self, flag: bool):

        self.__fg_border = flag

    @property
    def dir_img(self) -> pathlib.Path:

        return self.__dir_img

    @dir_img.setter
    def dir_img(self, p: pathlib.Path) -> None:

        if p is not None and not os.path.exists(p):
            raise Exception('Directory %s does not exist' % p)

        self.__dir_img = p

    @property
    def dir_trimap(self) -> pathlib.Path:

        return self.__dir_trimap

    @dir_trimap.setter
    def dir_trimap(self, p: pathlib.Path) -> None:

        if p is not None and not os.path.exists(p):
            raise Exception('Directory %s does not exist' % p)

        self.__dir_trimap = p

    @property
    def dir_xml(self) -> pathlib.Path:

        return self.__dir_xml

    @dir_xml.setter
    def dir_xml(self, p: pathlib.Path) -> None:

        if p is not None and not os.path.exists(p):
            raise Exception('Directory %s does not exist' % p)

        self.__dir_xml = p

    @property
    def file_training(self) -> pathlib.Path:

        return self.__training_txt

    @file_training.setter
    def file_training(self, f: pathlib.Path) -> None:

        if f is not None and not os.path.isfile(f):
            raise Exception('Path %s is not related to regular file' % f)

        self.__training_txt = f

    @property
    def file_test(self) -> pathlib.Path:

        return self.__test_txt

    @file_test.setter
    def file_test(self, f: pathlib.Path) -> None:

        if f is not None and not os.path.isfile(f):
            raise Exception('Path %s is not related to regular file' % f)

        self.__test_txt = f

    @property
    def mode_imread(self) -> OPENCV_IMREAD:

        return self.__opencv_imread

    @mode_imread.setter
    def mode_imread(self, f: OPENCV_IMREAD):

        self.__opencv_imread = f

    @property
    def output_dir(self) -> pathlib.Path:

        return self.__output_dir

    @output_dir.setter
    def output_dir(self, p: pathlib.Path):

        self.__output_dir = p

    @property
    def img_transformation(self) -> ImgTransformation:

        return self.__img_transformation

    @img_transformation.setter
    def img_transformation(self, f: ImgTransformation):

        self.__img_transformation = f

    @property
    def img_centered(self) -> bool:

        return self.__img_centered

    @img_centered.setter
    def img_centered(self, f: bool):

        self.__img_centered = f

    @property
    def img_shape(self) -> list:

        return self.__img_shape

    @img_shape.setter
    def img_shape(self, shape: list):

        self.__img_shape = shape

    @img_shape.deleter
    def img_shape(self):

        del self.__img_shape
        self.__img_shape = None

    def scaleImgsValues(self, v: float):

        for cat in self.__cats_training:
            cat.scale(v)

        for dog in self.__dogs_training:
            dog.scale(v)

    def normalizeImgs(self, norm_type: OPENCV_NORM):

        if not self.__cats_training:
            print('Warning: list of cats is empty')
            return

        if not self.__dogs_training:
            print('Warning: list of dogs is empty')
            return

        for cat in self.__cats_training:
            cat.normalize(norm_type.value)

        for dog in self.__dogs_training:
            dog.normalize(norm_type.value)

    def loadImg(self) -> None:

        if self.dir_img is None:
            raise Exception('Directory of pet images is not set')

        if self.dir_trimap is None:
            raise Exception('Directory of pet trimaps is not set')

        files_img = glob.glob(os.path.join(self.dir_img, '*.jpg'))

        for img in files_img:
            base = os.path.splitext(os.path.basename(img))[0]
            trimap = os.path.join(self.dir_trimap, base + '.png')

            obj_pet = ObjPet()
            obj_pet.loadImg(img, trimap, name=base, flags=self.mode_imread)

            if obj_pet.family == PetFamily.CAT:
                self.__cats_training.append(obj_pet)
            else:
                self.__dogs_training.append(obj_pet)

    def loadXml(self) -> None:

        if self.dir_xml is None:
            raise Exception('Directory of xmls is not set')

        if self.dir_img is None:
            raise Exception('Directory of pet images is not set')

        if self.dir_trimap is None:
            raise Exception('Directory of pet trimaps is not set')

        mlsys.FUNCTION_TRACE_BEGIN()

        files_xml = glob.glob(os.path.join(self.dir_xml, '*.xml'))

        for xml in files_xml:
            obj_pet = ObjPet()
            obj_pet.loadXml(xml_file=xml, dir_img=self.dir_img, dir_trimap=self.dir_trimap, flags=self.mode_imread)

            if obj_pet.family is PetFamily.CAT:
                self.__cats_training.append(obj_pet)
            else:
                self.__dogs_training.append(obj_pet)

        mlsys.FUNCTION_TRACE_END()

    def __createDatasetFromTxtFile(self, cats: list, dogs: list, file_dataset: pathlib.Path) -> None:

        if self.dir_xml is None:
            raise Exception('Directory of xmls is not set')

        if self.dir_img is None:
            raise Exception('Directory of pet images is not set')

        if self.dir_trimap is None:
            raise Exception('Directory of pet trimaps is not set')

        mlsys.FUNCTION_TRACE_BEGIN()

        csv_file = open(file_dataset)
        csv_reader = csv.reader(csv_file, delimiter=' ')

        for row in csv_reader:
            name = row[0]

            xml_file = os.path.join(self.dir_xml, name + '.xml')
            trimap_file = os.path.join(self.dir_trimap, name + '.png')

            if os.path.isfile(xml_file):
                obj_pet = ObjPet()
                obj_pet.loadXml(xml_file, self.dir_img, self.dir_trimap, flags=self.mode_imread)
            elif os.path.isfile(trimap_file):
                img_file = os.path.join(self.dir_img, name + '.jpg')

                obj_pet = ObjPet()
                obj_pet.loadImg(img_file, trimap_file, name=name, flags=self.mode_imread)
            else:
                print('%s excluded from dataset' % name)
                continue

            if obj_pet.family == PetFamily.CAT:
                cats.append(obj_pet)
            else:
                dogs.append(obj_pet)

        if self.verbose:
            ncats = len(cats)
            ndogs = len(dogs)
            npets = ncats + ndogs

            pcats = ncats / npets * 100
            pdogs = ndogs / npets * 100
            print('Dataset contains:')
            print(' samples %d \n samples+ (cat) %5d (%.2f %%) \n samples- (dog) %5d (%.2f %%)' % (npets, ncats, pcats, ndogs, pdogs))

        mlsys.FUNCTION_TRACE_END()

    def loadTrainingDataset(self) -> None:

        if self.file_training is None:
            raise Exception('File of training samples is not set')

        mlsys.FUNCTION_TRACE_BEGIN()
        self.__createDatasetFromTxtFile(self.__cats_training, self.__dogs_training, self.file_training)
        mlsys.FUNCTION_TRACE_END()

    def loadTestDataset(self) -> None:

        if self.file_training is None:
            raise Exception('File of test samples is not set')

        mlsys.FUNCTION_TRACE_BEGIN()
        self.__createDatasetFromTxtFile(self.__cats_test, self.__dogs_test, self.file_test)
        mlsys.FUNCTION_TRACE_END()

    def __mkdirOutput(self, dataset_type: DatasetType) -> pathlib.Path:

        if self.output_dir is not None:
            output_dir = os.path.join(self.output_dir, dataset_type.value)
        else:
            output_dir = dataset_type.value

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        return output_dir

    def __imgWrite(self, img: np.ndarray, name: str, ext: FORMAT, output_dir=''):

        p = name + '.%s' % ext.value
        if self.output_dir is not None or self.output_dir != '':
            p = os.path.join(output_dir, p)
        opencv.imwrite(p, img)

    def __scaleImg(self, img: np.ndarray, shape: list) -> np.ndarray:

        if img.shape[0] / img.shape[1] >= shape[0] / shape[1]:
            ratio = shape[0] / img.shape[0]
        else:
            ratio = shape[1] / img.shape[1]

        img_resized = opencv.resize(img, None, fx=ratio, fy=ratio)

        return img_resized

    def __fitImgShape(self, img_input: np.ndarray, shape) -> np.ndarray:

        img_colored = imgIsColored(img_input)

        if img_colored:
            tmp_img = np.zeros(shape=(shape[0], shape[1], img_input.shape[2]), dtype=img_input.dtype)
        else:
            tmp_img = np.zeros(shape=shape, dtype=img_input.dtype)

        if self.img_transformation != ImgTransformation.FIT_MAX:
            img = self.__scaleImg(img_input, shape)
        else:
            img = img_input

        if self.img_centered:
            x = int(shape[0] / 2. - img.shape[0] / 2.)
            y = int(shape[1] / 2. - img.shape[1] / 2.)
            if imgIsColored(img):
                for i in range(img.shape[2]):
                    tmp_img[x:(x + img.shape[0]), y:(y + img.shape[1]), i] = img[:, :, i]
            else:
                tmp_img[x:(x + img.shape[0]), y:(y + img.shape[1])] = img[:, :]
        else:
            tmp_img[0:img.shape[0], 0:img.shape[1]] = img[:, :]

        # UGLY hotfix
        if self.img_transformation != ImgTransformation.FIT_MAX:
            del img

        return tmp_img

    def __determineShapeStat(self, cats: list, dogs: list) -> list:

        shape = [0, 0]
        row_dims = cats[-1][0]
        col_dims = cats[-1][1]

        # TODO one class
        if dogs:
            row_dims.extend(dogs[-1][0])
            col_dims.extend(dogs[-1][1])

        np_rows = np.array(row_dims)
        np_cols = np.array(col_dims)

        if self.img_transformation.value < 2:
            shape[0] = int(np.mean(np_rows))
            shape[1] = int(np.mean(np_cols))
        elif self.img_transformation.value < 3:
            shape[0] = int(np.median(np_rows))
            shape[1] = int(np.median(np_cols))
        elif self.img_transformation.value < 4:
            shape[0] = int(np.min(np_rows))
            shape[1] = int(np.min(np_cols))
        else:
            shape[0] = int(np.max(np_rows))
            shape[1] = int(np.max(np_cols))

        del np_rows, np_cols

        return shape

    def __createLabelList(self, ncats, ndogs) -> list:

        if ncats > 0:
            labels_cat = [PetFamily.CAT.value] * ncats
        else:
            labels_cat = []
        if ndogs > 0:
            labels_dog = [PetFamily.DOG.value] * ndogs
        else:
            labels_dog = []
        return labels_cat + labels_dog

    def __getImg_ListArray(self, pets, get_mask=False, get_dims=False) -> (list, Union[list, None]):

        dataset = []
        list_rows = []
        list_cols = []

        mlsys.FUNCTION_TRACE_BEGIN()
        for pet in pets:
            pet.fg_include_border = self.fg_include_border
            out = pet.getImg(foreground=self.extract_foreground, roi=self.img_roi, get_mask=get_mask)
            if get_mask:
                img = out[0]
                mask = out[1]
                dataset.append([img, pet.name, mask])
            else:
                img = out
                dataset.append([img, pet.name])

            if get_dims:
                list_rows.append(img.shape[0])
                list_cols.append(img.shape[1])
        mlsys.FUNCTION_TRACE_END()

        if get_dims:
            return dataset, [list_rows, list_cols]
        else:
            return dataset

    # TODO rename writeImgs
    def __writeImgs(self, cats: list, dogs: list, ext: FORMAT, output_dir: pathlib.Path, shape=None):

        for cat in cats:
            if shape is not None:
                img_fitted = self.__fitImgShape(cat[0], shape)
                self.__imgWrite(img_fitted, cat[1], ext, output_dir)

                del img_fitted
            else:
                self.__imgWrite(cat[0], cat[1], ext, output_dir)

        for dog in dogs:
            if shape is not None:
                img_fitted = self.__fitImgShape(dog[0], shape)
                self.__imgWrite(img_fitted, dog[1], ext, output_dir)

                del img_fitted
            else:
                self.__imgWrite(dog[0], dog[1], ext, output_dir)

    def __saveImgs_ImageFileFormat(self, cats: list, dogs: list, ext: FORMAT, output_dir: pathlib.Path):

        mlsys.FUNCTION_TRACE_BEGIN()

        if self.img_transformation.value > -1 and not self.img_shape:
            get_dims = True
        else:
            get_dims = False

        if self.verbose:
            print('File format: %s' % ext.name)
            print('Output directory: %s' % pathlib.Path(output_dir).absolute())
            print('Centered scene: %r' % self.img_centered)

        out_cats = self.__getImg_ListArray(cats, get_dims=get_dims)
        out_dogs = self.__getImg_ListArray(dogs, get_dims=get_dims)

        if get_dims:
            self.img_shape = self.__determineShapeStat(out_cats, out_dogs)
            img_cats = out_cats[0]
            img_dogs = out_dogs[0]
            shape = self.img_shape
        else:
            img_cats = out_cats
            img_dogs = out_dogs
            if self.img_transformation.value > -1:
                shape = self.img_shape
            else:
                shape = None

        if self.verbose and shape is not None:
            print('Image transformation: %s to dims=[%d, %d]' % (self.img_transformation.name, shape[0], shape[1]))

        self.__writeImgs(img_cats, img_dogs, ext, output_dir, shape)
        mlsys.FUNCTION_TRACE_END()

    def __getImg_NumpyArray(self, cats: list, dogs: list) -> (np.ndarray, np.ndarray):

        mlsys.FUNCTION_TRACE_BEGIN()
        if self.img_transformation.value > -1 and not self.img_shape:
            get_dims = True
        else:
            get_dims = False

        out_cats = self.__getImg_ListArray(cats, get_dims=get_dims)
        out_dogs = self.__getImg_ListArray(dogs, get_dims=get_dims)

        if get_dims:
            self.img_shape = self.__determineShapeStat(out_cats, out_dogs)
            img_cats = out_cats[0]
            img_dogs = out_dogs[0]
            shape = self.img_shape
        else:
            img_cats = out_cats
            img_dogs = out_dogs
            if self.img_transformation.value > -1:
                shape = self.img_shape
            else:
                shape = None

        if self.verbose and shape is not None:
            print('Transforming source image: %s to dims=[%d, %d]' % (self.img_transformation.name, shape[0], shape[1]))

        np_pet = len(img_cats) + len(img_dogs)
        rows = np.empty(shape=(np_pet, shape[0] * shape[1]))

        idx = 0
        for cat in img_cats:
            if shape is not None:
                img_fitted = self.__fitImgShape(cat[0], shape)
                img_reshaped = img_fitted.reshape(-1)
                rows[idx] = img_reshaped

                del img_fitted, img_reshaped
            else:
                img_reshaped = cat[0].reshape(-1)
                rows[idx] = img_reshaped

                del img_reshaped

            idx += 1

        for dog in img_dogs:
            if shape is not None:
                img_fitted = self.__fitImgShape(dog[0], shape)
                img_reshaped = img_fitted.reshape(-1)
                rows[idx] = img_reshaped

                del img_fitted, img_reshaped
            else:
                img_reshaped = dog[0].reshape(-1)
                rows[idx] = img_reshaped

                del img_reshaped

            idx += 1

        # create labels related to training dataset
        labels = self.__createLabelList(len(img_cats), len(img_dogs))
        np_labels = np.array(labels, dtype=np.float)

        del out_cats, out_dogs
        del labels

        mlsys.FUNCTION_TRACE_END()

        return rows, np_labels

    def __saveImg_HDF5(self, cats: list, dogs: list, viewer):

        # TODO fit all time
        mlsys.FUNCTION_TRACE_BEGIN()

        if self.verbose:
            print('Output directory: %s' % viewer.output_dir)
            print('Centered scene: %r' % self.img_centered)

        # save flatten images to HDF5
        rows, labels = self.__getImg_NumpyArray(cats, dogs)
        viewer.feature_type = FeaturesType.DENSE
        viewer.save(rows, labels)

        del rows, labels

        mlsys.FUNCTION_TRACE_END()

    def __extractFeatures_getDescriptors(self, pets: list, feature_extractor: VisualFeatureExtractor, shape=None) -> tuple[np.ndarray, ...]:

        mlsys.FUNCTION_TRACE_BEGIN()
        desc_pet = []
        ndesc_pet = []

        for pet in pets:
            pet_img = pet[0]
            pet_mask = pet[2]

            if shape is not None:
                img_fitted = self.__fitImgShape(pet_img, shape)
                mask_fitted = self.__fitImgShape(pet_mask, shape)
                desc = feature_extractor.fit(img_fitted, mask_fitted)
                del img_fitted, mask_fitted
            else:
                desc = feature_extractor.fit(pet_img, pet_mask)

            if desc.shape[0] > 0:
                desc_pet.append(desc)
                ndesc_pet.append(desc.shape[0])

        # convert to numpy array
        if len(desc_pet) > 0:
            np_desc_pet = np.concatenate(desc_pet, axis=0)
            np_ndesc_pet = np.array(ndesc_pet)
        else:
            np_desc_pet = np.empty(shape=(0,))
            np_ndesc_pet = np.empty(shape=(0,))
        mlsys.FUNCTION_TRACE_END()

        return np_desc_pet, np_ndesc_pet

    def __extractFeatures_Train_VisualDictionary(self, descriptors: np.ndarray):

        if self.__cbtrained:
            return

        mlsys.FUNCTION_TRACE_BEGIN()
        del self.__visual_dictionary  # TODO reset?
        self.__visual_dictionary = VisualDictionary(verbose=self.verbose)

        self.__visual_dictionary.training_features = descriptors
        self.__visual_dictionary.codebook_size = self.xfeature_codebook_size
        self.__visual_dictionary.codebook_normalize = self.xfeature_codebook_normalize
        self.__visual_dictionary.train()  # TODO or fit?

        self.__cbtrained = True
        mlsys.FUNCTION_TRACE_END()

    def __extractFeatures_Fit(self, desc_pet: np.ndarray, ndesc_pet: np.ndarray) -> list:

        if desc_pet is None or ndesc_pet is None:
            return []

        mlsys.FUNCTION_TRACE_BEGIN()
        if not self.__cbtrained:
            self.__extractFeatures_Train_VisualDictionary()

        rows = []

        start = 0
        for i in range(len(ndesc_pet)):
            end = start + ndesc_pet[i]
            feature_pet = self.__visual_dictionary.fit(desc_pet[start:end])
            rows.append(feature_pet)
        mlsys.FUNCTION_TRACE_END()

        return rows

    def __extractFeatures_TrainFit(self, desc_cat: np.ndarray, ndesc_cat: np.ndarray,
                                         desc_dog: np.ndarray, ndesc_dog: np.ndarray) -> list:
        mlsys.FUNCTION_TRACE_BEGIN()

        descriptors = vstackNumpyArrays((desc_cat, desc_dog))
        self.__extractFeatures_Train_VisualDictionary(descriptors)
        del descriptors

        features_cat = self.__extractFeatures_Fit(desc_cat, ndesc_cat)
        features_dog = self.__extractFeatures_Fit(desc_dog, ndesc_dog)
        rows = features_cat + features_dog
        mlsys.FUNCTION_TRACE_END()

        return rows

    def __extractFeatures_Get(self, cats: list, dogs: list) -> (list, list):

        mlsys.FUNCTION_TRACE_BEGIN()
        feature_extractor = VisualFeatureExtractor(self.xfeature_type, self.verbose)

        if self.xfeature_type == FeatureExtractorType.SURF or FeatureExtractorType.DEFAULT:
            feature_extractor.SURF_hessianThreshold = self.SURF_hessianThreshold
            feature_extractor.SURF_nOctaves = self.SURF_nOctaves
            feature_extractor.SURF_nOctaveLayers = self.SURF_nOctaveLayers
            feature_extractor.SURF_extended = self.SURF_extended
            feature_extractor.SURF_upright = self.SURF_upright
        elif self.xfeature_type == FeatureExtractorType.SIFT:
            feature_extractor.SIFT_nfeatures = self.SIFT_nfeatures
            feature_extractor.SIFT_nOctaveLayers = self.SIFT_nOctaveLayers
            feature_extractor.SIFT_contrastThreshold = self.SIFT_contrastThreshold
            feature_extractor.SIFT_edgeThreshold = self.SIFT_edgeThreshold
            feature_extractor.SIFT_sigma = self.SIFT_sigma
        elif self.xfeature_type == FeatureExtractorType.ORB:
            feature_extractor.ORB_nfeatures = self.ORB_nfeatures
            feature_extractor.ORB_scaleFactor = self.ORB_scaleFactor
            feature_extractor.ORB_nlevels = self.ORB_nlevels
            feature_extractor.ORB_edgeThreshold = self.ORB_edgeThreshold
            feature_extractor.ORB_firstLevel = self.ORB_firstLevel
            feature_extractor.ORB_WTA_K = self.ORB_WTA_K
            feature_extractor.ORB_scoreType = self.ORB_scoreType
            feature_extractor.ORB_patchSize = self.ORB_patchSize
            feature_extractor.ORB_fastThreshold = self.ORB_fastThreshold
        else:
            raise Exception('Not supported type of feature extractor')

        if self.img_transformation.value > -1 and not self.img_shape:
            get_dims = True
        else:
            get_dims = False

        out_cats = self.__getImg_ListArray(cats, get_dims=get_dims, get_mask=True)
        out_dogs = self.__getImg_ListArray(dogs, get_dims=get_dims, get_mask=True)

        if get_dims:
            self.img_shape = self.__determineShapeStat(out_cats, out_dogs)
            img_cats = out_cats[0]
            img_dogs = out_dogs[0]
            shape = self.img_shape
        else:
            img_cats = out_cats
            img_dogs = out_dogs
            if self.img_transformation.value > -1:
                shape = self.img_shape
            else:
                shape = None

        if self.verbose and shape is not None:
            if shape is not None:
                print('Image transformation: %s to dims=[%d, %d]' % (self.img_transformation.name, shape[0], shape[1]))
            print('Feature extrator type: %s' % self.xfeature_type.name)  # TODO print params

        # obtain descriptors related to images
        desc_cat, ndesc_cat = self.__extractFeatures_getDescriptors(img_cats, feature_extractor, shape)
        desc_dog, ndesc_dog = self.__extractFeatures_getDescriptors(img_dogs, feature_extractor, shape)

        # extract features
        rows = self.__extractFeatures_TrainFit(desc_cat, ndesc_cat, desc_dog, ndesc_dog)
        # create list of labels
        labels = self.__createLabelList(ndesc_cat.shape[0], ndesc_dog.shape[0])
        np_labels = np.array(labels, dtype=np.float)

        # clean memory
        del desc_cat, ndesc_cat
        del desc_dog, ndesc_dog
        del out_cats, out_dogs
        del labels

        mlsys.FUNCTION_TRACE_END()

        return rows, np_labels

    def __extractFeatures_Save(self, cats: list, dogs: list, viewer):

        mlsys.FUNCTION_TRACE_BEGIN()

        # save result to HDF5
        rows, labels = self.__extractFeatures_Get(cats, dogs)
        viewer.output_dir = self.output_dir
        viewer.save(rows, labels, mat_type=FeaturesType.SPARSE)

        mlsys.FUNCTION_TRACE_END()

    def getTrainingDataset(self) -> (Union[list, np.ndarray], np.ndarray):

        mlsys.FUNCTION_TRACE_BEGIN()
        del self.img_shape
        self.img_shape = None

        if self.extract_features:
            self.__cbtrained = False
            rows, labels = self.__extractFeatures_Get(self.__cats_training, self.__dogs_training)
        else:
            rows, labels = self.__getImg_NumpyArray(self.__cats_training, self.__dogs_training)
        mlsys.FUNCTION_TRACE_END()

        return rows, labels

    def getTestDataset(self) -> (Union[list, np.ndarray], np.ndarray):

        mlsys.FUNCTION_TRACE_BEGIN()
        if self.extract_features:
            rows, labels = self.__extractFeatures_Get(self.__cats_test, self.__dogs_test)
        else:
            rows, labels = self.__getImg_NumpyArray(self.__cats_test, self.__dogs_test)
        mlsys.FUNCTION_TRACE_END()

        return rows, labels

    def saveTrainingDataset(self, ext=FORMAT.HDF5):

        mlsys.FUNCTION_TRACE_BEGIN()
        if self.img_shape is not None: del self.img_shape
        self.img_shape = None

        if self.extract_features or ext == FORMAT.HDF5:
            viewer_h5 = ViewerHDF5()
            viewer_h5.output_dir = self.output_dir
            viewer_h5.basename = DatasetType.TRAINING.value

            if self.extract_features:
                self.__cbtrained = False
                self.__extractFeatures_Save(self.__cats_training, self.__dogs_training, viewer_h5)
            else:
                self.__saveImg_HDF5(self.__cats_training, self.__dogs_training, viewer_h5)

            del viewer_h5
        else:
            output_dir = self.__mkdirOutput(DatasetType.TRAINING)
            self.__saveImgs_ImageFileFormat(self.__cats_training, self.__dogs_training, ext, output_dir)
        mlsys.FUNCTION_TRACE_END()

    def saveTestDataset(self, ext=FORMAT.HDF5):

        mlsys.FUNCTION_TRACE_BEGIN()
        if self.extract_features or ext == FORMAT.HDF5:
            viewer_h5 = ViewerHDF5()
            viewer_h5.output_dir = self.output_dir
            viewer_h5.basename = DatasetType.TEST.value

            if self.extract_features:
                self.__extractFeatures_Save(self.__cats_test, self.__dogs_test, viewer_h5)
            else:
                self.__saveImg_HDF5(self.__cats_test, self.__dogs_test, viewer_h5)

            del viewer_h5
        else:
            output_dir = self.__mkdirOutput(DatasetType.TEST)
            self.__saveImgs_ImageFileFormat(self.__cats_test, self.__dogs_test, ext, output_dir)
        mlsys.FUNCTION_TRACE_END()

    def save(self, ext=FORMAT.HDF5):

        mlsys.FUNCTION_TRACE_BEGIN()
        self.saveTrainingDataset(ext)
        self.saveTestDataset(ext)
        mlsys.FUNCTION_TRACE_END()

# TODO feature extraction setting params
# TODO SVMLight format, PETSc bin?
# TODO verbose (partially)
# TODO set name of test file
# TODO check if dataset is loaded
# TODO save sparse dataset compatible with matlab
