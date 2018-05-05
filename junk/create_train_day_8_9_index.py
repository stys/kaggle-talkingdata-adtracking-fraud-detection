import numpy as np
from lib.columns import DataFrameCols

if __name__ == '__main__':
    workdir = '../data/columns'
    dfc = DataFrameCols(workdir)

    day_col = dfc.load_column('day')
    is_attributed_col = dfc.load_column('is_attributed')
    index = np.where((is_attributed_col >= 0) & (day_col > 7))[0]

    dfc.write_index('days_8_9_attributed', index)
