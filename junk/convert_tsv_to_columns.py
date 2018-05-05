import numpy as np
import pandas as pd

from lib.columns import DataFrameCols
from lib.utils import makedirs

if __name__ == '__main__':
    df = pd.read_csv('../data/train_test_merged/train_test_merged.tsv', sep='\t', parse_dates=['click_time'])

    rename_columns = {}
    for col in df.columns:
        if col.startswith('# '):
            rename_columns[col] = col[2:]
    df.rename(columns=rename_columns, inplace=True)
    df.sort_values(by=['id'], inplace=True)

    dtypes = {
        'id': 'uint32',
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'click_id': 'int32',
        'click_id_submission': 'int32',
        'is_attributed': 'int8'
    }

    test_dir = '../data/columns'
    makedirs(test_dir)

    dfc = DataFrameCols(test_dir)
    for col, dtype in dtypes.items():
        print(col, dtype)
        dfc.write_column(col, df[col].astype(dtype).values)

    dfc.write_column('day', pd.to_datetime(df['click_time']).dt.day.astype('uint8').values)
    dfc.write_column('hour', pd.to_datetime(df['click_time']).dt.hour.astype('uint8').values)
    dfc.write_column('epoch', (df['click_time'].astype(np.int64) // 10 ** 9).values)
