from os.path import join, isfile

import sys
import ast
import numpy as np
import pandas as pd

from argparse import ArgumentParser


class DataFrameCols(object):
    COL_EXT = '.bin'
    IDX_EXT = '.idx'
    META = 'meta'

    def __init__(self, workdir):
        self.workdir = workdir
        self.meta = DataFrameCols.read_meta(workdir)

    @staticmethod
    def read_meta(workdir):
        meta_file = join(workdir, DataFrameCols.META)
        if not isfile(meta_file):
            return {}
        else:
            with open(join(workdir, DataFrameCols.META), 'r') as fmeta:
                return ast.literal_eval(fmeta.read())

    def _write_meta(self):
        with open(join(self.workdir, DataFrameCols.META), 'w') as fmeta:
            fmeta.write(str(self.meta))

    def load_column(self, col, arange=None, index=None):
        arr = np.fromfile(join(self.workdir, col + DataFrameCols.COL_EXT), dtype=self.meta[col])
        if arange is not None:
            start_index = arange[0]
            end_index = arange[1]
            return arr[start_index:end_index]
        elif index is not None:
            return arr[index]
        else:
            return arr

    def write_column(self, name, arr, arange=None, index=None):
        if name in self.meta:
            assert self.meta[name] == arr.dtype
        else:
            self.meta[name] = arr.dtype.str

        if arange is not None:
            start_index = arange[0]
            end_index = arange[0]
            arr[start_index:end_index].tofile(join(self.workdir, name + DataFrameCols.COL_EXT))
        else:
            arr[index].tofile(join(self.workdir, name + DataFrameCols.COL_EXT))

        self._write_meta()

    def load_df(self, columns=None, arange=None, index=None):
        data = dict()
        columns = columns or self.meta.keys()
        for col in columns:
            data[col] = self.load_column(col, arange, index)
        return pd.DataFrame(data=data)

    def write_df(self, df, arange=None, index=None):
        for i, col in enumerate(df.columns):
            self.write_column(col, df[col].values, arange, index)

    def load_index(self, name):
        return np.fromfile(join(self.workdir, name + DataFrameCols.IDX_EXT), dtype=np.uint32)

    def write_index(self, name, arr):
        arr.astype(np.uint32).tofile(join(self.workdir, name + DataFrameCols.IDX_EXT))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path', default='.')
    parser.add_argument('-f', '--fields', nargs='+', default=None)
    parser.add_argument('--range-start', type=int, default=None)
    parser.add_argument('--range-end', type=int, default=None)
    parser.add_argument('--index', default=None)
    args = parser.parse_args()

    dfc = DataFrameCols(args.path)
    df = dfc.load_df(columns=args.fields, arange=(args.range_start, args.range_end))
    df.to_csv(sys.stdout, header=True, index=False, sep='\t')
