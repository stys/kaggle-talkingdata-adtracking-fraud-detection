import numpy as np
from lib.columns import DataFrameCols

if __name__ == '__main__':
    workdir = '../data/columns'
    dfc = DataFrameCols(workdir)
    day_col = dfc.load_column('day')
    hour_col = dfc.load_column('hour')
    is_attributed_col = dfc.load_column('is_attributed')

    hours = np.unique(hour_col, return_counts=True)[0]

    for h in hours:
        all = np.where((day_col == 9) & (hour_col == h))[0].shape[0]
        attributed = np.where((day_col == 9) & (hour_col == h) & (is_attributed_col >= 0))[0].shape[0]

        print(h, all, attributed)

# hour all attributed
# 0 3318301 3318301
# 1 3082862 3082862
# 2 3068887 3068887
# 3 3351149 3351149
# 4 4032691 4032691
# 5 3671741 3671741
# 6 3570940 3570940
# 7 3186240 3186240
# 8 2804701 2804701
# 9 2986204 2986204
# 10 3304199 3304199
# 11 3347741 3347741
# 12 3363917 3363917
# 13 3457523 3457523
# 14 3443348 3443283      !!! hour 14 has small fraction of not attributed events
# 15 3026679 3026111
# 16 2495595 447
# 17 1265180 0
# 18 762056 0
# 19 526096 0
# 20 432411 0
# 21 571504 0
# 22 1325626 0
# 23 2423959 0