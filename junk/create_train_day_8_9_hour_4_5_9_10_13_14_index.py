import numpy as np
from lib.columns import DataFrameCols

if __name__ == '__main__':
    workdir = '../data/columns'
    dfc = DataFrameCols(workdir)

    day_col = dfc.load_column('day')
    hour_col = dfc.load_column('hour')
    is_attributed_col = dfc.load_column('is_attributed')

    hidx = (hour_col == 4) | (hour_col == 5) | (hour_col == 9) | (hour_col == 10) | (hour_col == 13) | (hour_col == 14)
    index = np.where((is_attributed_col >= 0) & (day_col > 7) & hidx)[0]

    dfc.write_index('days_8_9_hours_4_5_9_10_13_14_attributed', index)
