import os
import pathlib

from mlpet.data.dataset import Dataset as OxfordIIITDataset
from mlpet.utils.types import *


def main() -> None:

    data_dir = pathlib.Path(os.environ['DATA_DIR'])
    output_dir = pathlib.Path(os.environ['OUTPUT_DIR'])

    dir_annotation = os.path.join(data_dir, 'annotations')
    dir_img = os.path.join(data_dir, 'images')
    dir_trimap = os.path.join(dir_annotation, 'trimaps')
    dir_xml = os.path.join(dir_annotation, 'xmls')

    iiit_dataset = OxfordIIITDataset(verbose=True)

    # setting directories of images, trimaps and xml files
    iiit_dataset.dir_img = dir_img
    iiit_dataset.dir_trimap = dir_trimap
    iiit_dataset.dir_xml = dir_xml

    # setting parameters for exporting images
    iiit_dataset.img_roi = ROI.PET
    iiit_dataset.fg_include_border = False  # including border of pet
    iiit_dataset.mode_imread = OPENCV_IMREAD.GRAYSCALE  # reading images as grayscale

    # scale all images to median of dataset dimensions
    iiit_dataset.img_transformation = ImgTransformation.NONE
    iiit_dataset.img_centered = False
    iiit_dataset.extract_features = True
    iiit_dataset.xfeature_type = FeatureExtractorType.SIFT

    # load a training dataset
    iiit_dataset.file_training = os.path.join('data/csv/trainval_selected.txt')
    iiit_dataset.loadTrainingDataset()

    # load a test data set
    iiit_dataset.file_test = os.path.join('data/csv/test_selected.txt')
    iiit_dataset.loadTestDataset()

    for xtype in (FeatureExtractorType.SIFT, FeatureExtractorType.SURF, FeatureExtractorType.ORB):
        for vd_size in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            for norm in (False, True):

                print(f'feature_type={xtype}, vd_size={vd_size}, normalized={norm}')

                iiit_dataset.xfeature_codebook_size = vd_size
                iiit_dataset.xfeature_codebook_normalize = norm
                iiit_dataset.xfeature_type = xtype

                fn = f'{xtype.name.lower()}_{vd_size}'
                if norm: fn = f'{fn}_norm'
                fn = os.path.join(output_dir, fn)

                # save a test data set ti HDF5 file
                iiit_dataset.saveTrainingDataset(f'{fn}_training', ext=FORMAT.HDF5, mat_type=FeaturesType.DENSE)

                # save a test data set to HDF5 file
                iiit_dataset.saveTestDataset(f'{fn}_test', ext=FORMAT.HDF5, mat_type=FeaturesType.DENSE)


if __name__ == '__main__':

    main()
