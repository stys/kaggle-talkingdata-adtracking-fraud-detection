import numpy as np
from lib.columns import DataFrameCols

if __name__ == '__main__':
    workdir = '../data/columns'
    dfc = DataFrameCols(workdir)

    click_id_submission = dfc.load_column(col='click_id_submission')
    index = np.where(click_id_submission >= 0)[0].astype(np.uint32)
    dfc.write_index('submission', index)
