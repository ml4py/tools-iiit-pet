import os
import pathlib

import xml.dom.minidom as xml_dom

from typing import Union

# import utils
from mlpet.utils.functools import lazy_import
from mlpet.utils.types import PetFamily, OPENCV_NORM, OPENCV_IMREAD, ROI, imgIsColored

# lazy imports
_opencv = lazy_import('cv2')
_np = lazy_import('numpy')


class ObjPet(object):

    def __init__(self, img=None, family=PetFamily.NONE, trimap: _np.ndarray = None, roi_face: _np.ndarray = None,
                 name: str = None):

        self.img = img
        self.trimap = trimap
        self.roi_face = roi_face
        self.family = family
        self.name = name

        self.fg_include_border = True

    def __del__(self):

        del self.img
        del self.trimap
        self.family = PetFamily.NONE
        self.name = None

    @property
    def img(self) -> _np.ndarray:

        return self.__img

    @img.setter
    def img(self, img: _np.ndarray):

        self.__img = img

    @img.deleter
    def img(self):

        del self.__img

    @property
    def trimap(self) -> _np.ndarray:

        return self.__trimap

    @trimap.setter
    def trimap(self, img: _np.ndarray):

        if img is not None:
            if _np.min(img) == _np.max(img):
                raise Exception('Warning: trimap is not defined by source (%s)' % self.name)

        self.__trimap = img

    @trimap.deleter
    def trimap(self):

        del self.__trimap

    @property
    def roi_face(self) -> _np.ndarray:

        return self.__roi_face

    @roi_face.setter
    def roi_face(self, roi: _np.ndarray):

        if roi is not None and roi.dtype != _np.uint:
            raise Exception('ROI of pet head must be numpy.ndarray of integer type %s' % self.name)

        self.__roi_face = roi

    @roi_face.deleter
    def roi_face(self):

        del self.__roi_face

    @property
    def family(self) -> PetFamily:

        return self.__family

    @family.setter
    def family(self, f: PetFamily):

        self.__family = f

    @property
    def name(self) -> str:

        return self.__name

    @name.setter
    def name(self, name: str):

        self.__name = name

    @property
    def fg_include_border(self) -> bool:

        return self.__fg_include_border

    @fg_include_border.setter
    def fg_include_border(self, flag: bool):

        self.__fg_include_border = flag

    def getBackgroundMask(self) -> _np.ndarray:

        if self.trimap is None:
            raise Exception('Trimap is not set (%s)' % self.name)

        if self.fg_include_border:
            mask = self.trimap == 2
        else:
            mask = _np.logical_or(self.trimap == 2, self.trimap == 3)

        return 255 * mask.astype(_np.uint8)

    def getForegroundMask(self) -> _np.ndarray:

        if self.trimap is None:
            raise Exception('Trimap is not set (%s)' % self.name)

        return _opencv.bitwise_not(self.getBackgroundMask())

    def getForeground(self) -> _np.ndarray:

        mask = self.getForegroundMask()
        return self.getMaskedImg(mask)

    def getMaskedImg(self, mask: _np.ndarray) -> _np.ndarray:

        if imgIsColored(self.img):
            dst = _np.zeros(shape=self.img.shape, dtype=self.img.dtype)
            for i in range(0, self.img.shape[2]):
                dst[:, :, i] = _opencv.bitwise_and(self.img[:, :, i], mask)
            return dst
        else:
            return _opencv.bitwise_and(self.img, mask)

    def getFaceROI(self, foreground=False) -> (_np.ndarray, Union[_np.ndarray, None]):

        if self.roi_face is None:
            raise Exception('ROI of pet face is not set (%s)' % self.name)

        if foreground:
            mask = self.getForegroundMask()
            img = self.getMaskedImg(mask)
        else:
            img = self.img.copy()
            mask = None

        xmin = self.roi_face[0]
        xmax = self.roi_face[1]
        ymin = self.roi_face[2]
        ymax = self.roi_face[3]

        img = img[ymin:ymax, xmin:xmax]
        if mask is not None:
            mask = mask[ymin:ymax, xmin:xmax]
            return img, mask
        else:
            return img

    def getPetROI(self, foreground=False, get_mask=False) -> (_np.ndarray, Union[_np.ndarray, None]):

        if foreground:
            mask = self.getForegroundMask()
            img = self.getMaskedImg(mask)
        else:
            img = self.img.copy()
            mask = None

        idx = _opencv.findNonZero(self.getForegroundMask())
        rect = _opencv.boundingRect(idx)

        img = img[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        if get_mask:
            mask = mask[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
            return img, mask
        else:
            return img

    def __fitImgShape(self, img: _np.ndarray, shape, centered):

        if img.shape[0] < shape[0] or img.shape[1] < shape[1]:
            tmp_img = _np.zeros(shape=shape, dtype=img.dtype)

            if centered:
                x = int(shape[1] / 2. - img.shape[1] / 2.)
                y = int(shape[0] / 2. - img.shape[0] / 2.)
                if imgIsColored(img):
                    for i in range(img.shape[2]):
                        tmp_img[y:(y + img.shape[0]), x:(x + img.shape[1]), i] = img[:, :, i]
                else:
                    tmp_img[y:(y + img.shape[0]), x:(x + img.shape[1])] = img[:, :]
            else:
                tmp_img[0:img.shape[0], 0:img.shape[1]] = img[:, :]

            img = tmp_img
        else:
            img = img[0:shape[0], 0:shape[1]]

        return img

    def getImg(self, foreground=False, roi=ROI.ALL, shape=None, centered=False, get_mask=False) \
            -> (_np.ndarray, Union[_np.ndarray, None]):

        if roi == ROI.PET:
            out = self.getPetROI(foreground, get_mask)

            if get_mask:
                img = out[0]
                mask = out[1]
            else:
                img = out
        elif roi == ROI.FACE:
            # TODO get mask
            out = self.getFaceROI(foreground)
            if get_mask:
                img = out[0]
                mask = out[1]
            else:
                img = out
        else:
            img = self.img

        if shape is not None:
            img = self.__fitImgShape(img, shape, centered)
            if get_mask:
                mask = self.__fitImgShape(mask, shape, centered)

        if get_mask:
            return img, mask
        else:
            return img

    def scale(self, v: float):

        self.img /= v

    def normalize(self, norm_type: OPENCV_NORM = OPENCV_NORM.MINMAX):

        self.img = _opencv.normalize(self.img, None, alpha=0, beta=1, norm_type=norm_type, dtype=_opencv.CV_32F)

    def loadImg(self, file_img: pathlib.Path, file_trimap: pathlib.Path, name: str = None, flags=OPENCV_IMREAD.GRAYSCALE):

        if not os.path.exists(file_img):
            raise Exception('File image \'%s\' of pet is not exist' % file_img)

        if not os.path.exists(file_trimap):
            raise Exception('File trimap \'%s\' related to pet image is not exist' % file_trimap)

        # set object name and pet family
        if name is None or name == '':
            self.name = os.path.splitext(os.path.basename(file_trimap))[0]
        else:
            self.name = name

        # read source image of pet
        self.img = _opencv.imread(file_img, flags=flags.value)
        if self.img is None or self.img.size == 0:
            raise Exception('Image of pet is not loaded successfully (%s)' % file_img)

        # read trimap related to pet
        self.trimap = _opencv.imread(file_trimap, _opencv.IMREAD_GRAYSCALE)
        if self.trimap is None:
            raise Exception('Trimap related to pet image is not loaded successfully')

        self.family = PetFamily.CAT if self.name[0].isupper() else PetFamily.DOG

    def loadXml(self, xml_file: pathlib.Path, dir_img: pathlib.Path, dir_trimap: pathlib.Path,
                flags=OPENCV_IMREAD.GRAYSCALE):

        xdom = xml_dom.parse(xml_file)

        # determine path of source image
        file_img = xdom.getElementsByTagName('filename')[0].firstChild.data
        path_img = os.path.join(dir_img, file_img)

        # get basename of file related to input image
        name = os.path.splitext(file_img)[0]

        # determine path of trimap
        path_trimap = os.path.join(dir_trimap, name + '.png')

        # load images and set object name
        self.loadImg(path_img, path_trimap, name, flags)

        # bounding box related to pet face
        bndbox = xdom.getElementsByTagName('bndbox')[0]
        xmin = _np.int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
        xmax = _np.int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
        ymin = _np.int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
        ymax = _np.int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
        self.roi_face = _np.array([xmin, xmax, ymin, ymax], dtype=_np.uint)

        del xdom

# TODO function for returning mask of ROI
# TODO fit/scale to min, max, medium, mean
