import numpy as np
from lib.columns import DataFrameCols

if __name__ == '__main__':
    workdir = '../data/columns'
    dfc = DataFrameCols(workdir)

    submission_idx = dfc.load_index('submission')
    day_col = dfc.load_column('day', index=submission_idx)
    hour_col = dfc.load_column('hour', index=submission_idx)

    print('Submission days:')
    days = np.unique(day_col, return_counts=True)
    for (d, c) in zip(days[0], days[1]):
        print(d, c)

    print('Submission hours')
    hours = np.unique(hour_col, return_counts=True)
    for (h, c) in zip(hours[0], hours[1]):
        print(h, c)

# Submission days:
# 10 18790469
# Submission hours
# 4 3344125
# 5 2858427
# 6 381
# 9 2984808
# 10 3127993
# 11 413
# 13 3212566
# 14 3261257
# 15 499
