import os

from dataset import Dataset as OxfordIIITDataset
from Types import *

work_dir = '/Users/marek/Playground/Oxford-IIIT-Pet-Dataset'

dir_annotation = os.path.join(work_dir, 'annotations')
dir_img = os.path.join(work_dir, 'images')
dir_trimap = os.path.join(work_dir, 'trimap')
dir_xml = os.path.join(work_dir, 'xmls')

iiit_dataset = OxfordIIITDataset(verbose=True)

# setting directories of images, trimaps and xml files
iiit_dataset.dir_img = dir_img
iiit_dataset.dir_trimap = dir_trimap
iiit_dataset.dir_xml = dir_xml

# setting parameters for exporting images
iiit_dataset.img_roi = ROI.PET
iiit_dataset.fg_include_border = True  # including border of pet
iiit_dataset.mode_imread = OPENCV_IMREAD.GRAYSCALE  # reading images as grayscale
# scale all images to median of dataset dimensions
iiit_dataset.img_transformation = ImgTransformation.SCALE_MEDIAN
iiit_dataset.img_centered = False
iiit_dataset.extract_features = True
iiit_dataset.xfeature_type = FeatureExtractorType.SURF

# export dataaset
iiit_dataset.file_training = os.path.join(dir_annotation, 'trainval-subset.txt')
iiit_dataset.loadTrainingDataset()
iiit_dataset.saveTrainingDataset(FORMAT.HDF5)
