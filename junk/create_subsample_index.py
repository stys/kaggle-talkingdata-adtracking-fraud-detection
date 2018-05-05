import numpy as np
from lib.columns import DataFrameCols

if __name__ == '__main__':
    workdir = '../data/columns'
    dfc = DataFrameCols(workdir)

    is_attributed_col = dfc.load_column('is_attributed')
    subsample = np.random.choice([0, 1], size=is_attributed_col.shape[0], p=[0.5, 0.5])
    subsample_idx = np.where((is_attributed_col == 1) | ((is_attributed_col == 0) & (subsample == 1)))[0]

    print(subsample_idx.shape[0])
    dfc.write_index('subsample_not_attributed_50pct', subsample_idx)
