import numpy as np
from lib.columns import DataFrameCols

if __name__ == '__main__':
    workdir = '../data/columns'
    dfc = DataFrameCols(workdir)

    is_attributed_col = dfc.load_column('is_attributed')
    index = np.where((is_attributed_col >= 0))[0]

    print(index.shape[0])
    dfc.write_index('train', index)
