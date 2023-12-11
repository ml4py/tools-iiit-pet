import glob
import os.path
import random

from sklearn.model_selection import train_test_split
from mlpet.utils.functools import lazy_import

from mlpet.utils.types import OPENCV_IMREAD

# lazy imports
_opencv = lazy_import('cv2')


def main() -> None:

    dir_xml = '/scratch/data/public/oxford-iiit-pet/annotations/xmls'
    dir_trimaps = '/scratch/data/public/oxford-iiit-pet/annotations/trimaps'
    dir_output = '/scratch/apps/tools-iiit-pet/examples'

    lst_xml = glob.glob(f'{dir_xml}/*.xml')
    lst_xml = set((os.path.splitext(os.path.basename(fn))[0] for fn in lst_xml))
    print(len(lst_xml))

    lst_png = glob.glob(f'{dir_trimaps}/*.png')
    # exclude trimaps
    for png in lst_png:
        np_img = _opencv.imread(png, flags=OPENCV_IMREAD.GRAYSCALE.value)
        if np_img.min() == np_img.max():
            print(f'{png} excluded')
            lst_png.remove(png)

    lst_png = set((os.path.splitext(os.path.basename(fn))[0] for fn in lst_png))
    lst_xml = lst_xml.intersection(lst_png)

    lst_xml_dogs = []
    lst_xml_cats = []

    for fn in lst_xml:
        if str.isupper(fn[0]):
            lst_xml_cats.append(fn)
        else:
            lst_xml_dogs.append(fn)

    random.seed(42)
    random.shuffle(lst_xml_cats)
    random.shuffle(lst_xml_dogs)

    ncats = len(lst_xml_cats)
    selected_lst_xml_dog = random.choices(lst_xml_dogs, k=ncats)

    # create labels
    xml_pet = lst_xml_cats + selected_lst_xml_dog
    labels = ncats * [1] + ncats * [-1]

    xml_pet_train, xml_pet_test, _, _ = train_test_split(xml_pet, labels, test_size=0.10, random_state=42)

    csv_train = os.path.join(dir_output, 'trainval_selected.txt')
    with open(csv_train, 'w') as f:
        for xml in xml_pet_train: f.write(f'{xml}\n')

    csv_test = os.path.join(dir_output, 'test_selected.txt')
    with open(csv_test, 'w') as f:
        for xml in xml_pet_test: f.write(f'{xml}\n')


if __name__ == '__main__':
    main()
