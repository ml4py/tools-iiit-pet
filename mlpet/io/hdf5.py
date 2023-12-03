import os
import pathlib

import numpy as np
import h5py
import hdf5storage as h5storage

import mlpet.utils.sys as mlsys

from typing import Union
from mlpet.utils.types import FeaturesType


class ViewerHDF5:

    def __init__(self):

        self.output_dir = None
        self.basename = None
        self.feature_type = FeaturesType.SPARSE

    @property
    def output_dir(self) -> pathlib.Path:

        return self.__dir_output

    @output_dir.setter
    def output_dir(self, dir: pathlib.Path) -> None:

        self.__dir_output = dir

    @property
    def basename(self) -> str:

        return self.__filename

    @basename.setter
    def basename(self, f: str) -> None:

        self.__filename = f

    @property
    def feature_type(self) -> FeaturesType:

        return self.__feature_type

    @feature_type.setter
    def feature_type(self, f: FeaturesType) -> None:

        self.__feature_type = f

    @staticmethod
    def __determineAIJ(rows: list):

        ncols = 0
        a = np.empty(shape=(0,))
        i = []
        j = np.empty(shape=(0,), dtype=int)

        idx = 0
        for row in rows:
            i.append(idx)

            c = np.nonzero(row != 0)
            j = np.append(j, c)
            a = np.append(a, row[c])

            if len(row) > ncols: ncols = len(row)
            idx += np.shape(c)[1]
        i.append(idx)
        return a, i, j, ncols

    def __saveSparseFeaturesDataset(self, rows: list, labels: Union[list, np.ndarray], path: pathlib.Path) -> None:

        mlsys.FUNCTION_TRACE_BEGIN()
        a, i, j, ncols = self.__determineAIJ(rows)

        if self.output_dir is not None and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        h5_file = h5py.File(path, 'w')

        h5_file.create_dataset('y', data=labels)
        mat_group = h5_file.create_group('X')
        mat_group.attrs["MATLAB_sparse"] = ncols
        mat_group.create_dataset('data', data=a)
        mat_group.create_dataset('jc', data=i)
        mat_group.create_dataset('ir', data=j)

        h5_file.close()

        del a, i, j
        mlsys.FUNCTION_TRACE_END()

    @staticmethod
    def __saveDenseFeaturesDataset(rows: np.ndarray, labels: np.ndarray, path: pathlib.Path) -> None:

        mlsys.FUNCTION_TRACE_BEGIN()
        content = {'X': rows, 'y': labels}
        # this is extremely slow and memory consuming
        h5storage.write(content, filename=path,  matlab_compatible=True, store_python_metadata=False)
        mlsys.FUNCTION_TRACE_END()

    def save(self, rows: Union[list, np.ndarray], labels: Union[list, np.ndarray], mat_type=FeaturesType.DENSE) -> None:

        mlsys.FUNCTION_TRACE_BEGIN()
        if self.output_dir is not None and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        file = self.basename + '.h5'
        if self.output_dir is not None:
            path = os.path.join(self.output_dir, file)
        else:
            path = file

        if mat_type == FeaturesType.DENSE:
            self.__saveDenseFeaturesDataset(rows, labels, pathlib.Path(path))
        else:
            self.__saveSparseFeaturesDataset(rows, labels, pathlib.Path(path))
        mlsys.FUNCTION_TRACE_END()
