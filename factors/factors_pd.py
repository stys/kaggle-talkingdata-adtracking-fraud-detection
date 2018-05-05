import gc
import logging
from os.path import abspath, join as join_path

import numpy as np
import pandas as pd

from lib.project import project
from lib.columns import DataFrameCols
from lib.hocon import write_config, config2json
from lib.utils import makedirs


class Factors(object):

    def datetimes(self, df):
        df['day'] = pd.to_datetime(df['click_time']).dt.day.astype('uint8')
        df['hour'] = pd.to_datetime(df['click_time']).dt.hour.astype('uint8')
        return df

    def aggr(self, df, name, groupby, select, aggr, dtype, **other):
        """ Baris' aggregates
        https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977?scriptVersionId=3224614
        https://www.kaggle.com/aharless/kaggle-runnable-version-of-baris-kanber-s-lightgbm
        """
        grouped = df[groupby + [select]].groupby(groupby)[select]
        if aggr == 'count':
            count = grouped.count().reset_index().rename(columns={select: name})
            df = df.merge(count, on=groupby, how='left')
            df[name] = df[name].astype(dtype)
        if aggr == 'nunique':
            nunique = grouped.nunique().reset_index().rename(columns={select: name})
            df = df.merge(nunique, on=groupby, how='left')
            df[name] = df[name].astype(dtype)
        if aggr == 'mean':
            mean = grouped.mean().reset_index().rename(columns={select: name}).fillna(0)
            df = df.merge(mean, on=groupby, how='left')
            df[name] = df[name].astype(dtype)
        if aggr == 'var':
            var = grouped.var().reset_index().rename(columns={select: name}).fillna(0)
            df = df.merge(var, on=groupby, how='left')
            df[name] = df[name].astype(dtype)
        if aggr == 'cumcount':
            cumcount = grouped.cumcount()
            df[name] = cumcount.values
            df[name] = df[name].astype(dtype)

        del grouped
        gc.collect()

        return df

    def hash_id(self, df, name, groupby, num_bits=27, salt='salt', **other):
        d = (1 << num_bits)

        def hashfcn(row):
            if row['id'] % 1000 == 0:
                logging.info(row['id'])
            return hash(salt + '_'.join(map(str, [row[k] for k in groupby]))) % d

        df[name] = df.apply(hashfcn, axis=1, reduce=False).astype(np.uint32)
        return df

    def click_time(self, df, name, hash_id, reverse=False, num_bits=27, **other):
        """ Baris' time to prev/next click
        https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977?scriptVersionId=3224614
        https://www.kaggle.com/aharless/kaggle-runnable-version-of-baris-kanber-s-lightgbm
        """
        d = (1 << num_bits)

        epochs = df['epoch'].values
        ids = df[hash_id].values
        if reverse:
            ids = reversed(ids)
            epochs = reversed(epochs)

        unknown = np.iinfo(np.uint32).max
        buf = np.full(d, unknown, dtype=np.uint32)
        prev_click = np.full(df.shape[0], unknown, dtype=np.uint32)

        for i, (_id, t) in enumerate(zip(ids, epochs)):
            t_prev = buf[_id]
            buf[_id] = t
            if t_prev != unknown:
                if not reverse:
                    prev_click[i] = t - t_prev
                else:
                    prev_click[i] = t_prev - t

        if not reversed:
            df[name] = prev_click
        else:
            df[name] = np.flipud(prev_click)

        return df

    def click_time_no_hash(self, df, name, groupby, step=1, reverse=False, **other):
        """ Compute previous/next click time without hashing trick
        https://www.kaggle.com/asydorchuk/nextclick-calculation-without-hashing-trick
        """
        if not reverse:
            df[name] = df['epoch'] - df.groupby(groupby)['epoch'].shift(step).fillna(0)
        else:
            df[name] = df.groupby(groupby)['epoch'].shift(-step).fillna(3000000000) - df['epoch']

        return df

    def overflow_counter(self, df, name, hash_id, limit, reverse=False, num_bits=27, **other):
        d = (1 << num_bits)
        start_time = df['epoch'].iloc[0] if not reverse else df['epoch'].iloc[-1]

        counts = np.zeros(d, dtype=np.uint32)
        overflow_time = np.full(d, start_time, dtype=np.uint32)
        prev_overflow_time = np.full(d, start_time, dtype=np.uint32)

        dt1 = np.zeros(df.shape[0], dtype=np.uint32)
        dt2 = np.zeros(df.shape[0], dtype=np.uint32)
        levels = np.zeros(df.shape[0], dtype=np.uint32)

        ids = df[hash_id].values
        epochs = df['epoch'].values
        if reversed:
            ids = reversed(ids)
            epochs = reversed(epochs)

        for i, (_id, t0) in enumerate(zip(ids, epochs)):
            counts[_id] += 1
            level = counts[_id] % limit
            if level == 0:
                prev_overflow_time[_id] = overflow_time[_id]
                overflow_time[_id] = t0

            levels[i] = level
            if not reverse:
                dt1[i] = t0 - overflow_time[_id]
                dt2[i] = overflow_time[_id] - prev_overflow_time[_id]
            else:
                dt1[i] = overflow_time[_id] - t0
                dt2[i] = prev_overflow_time[_id] - overflow_time[_id]

        del counts, overflow_time, prev_overflow_time
        gc.collect()

        if not reverse:
            df[name + '_level'] = levels
            df[name + '_dt1'] = dt1
            df[name + '_dt2'] = dt2
        else:
            df[name + '_level'] = np.flipud(levels)
            df[name + '_dt1'] = np.flipud(dt1)
            df[name + '_dt2'] = np.flipud(dt2)

        return df


def main(conf):
    dump_dir = abspath(conf['factors_pd']['dump']['dir'])
    makedirs(dump_dir)

    col_dir = abspath(conf['factors_pd']['source'])
    dfc = DataFrameCols(col_dir)

    computer = Factors()
    for group in conf['factors_pd']['factors']:
        logging.info('Compute factors group: %s', group)
        for factor in conf['factors_pd']['factors'][group]:
            logging.info('Compute factor: %s', factor)
            spec = conf['factors_pd']['factors'][group][factor]
            df = dfc.load_df(['id'] + spec['columns'])
            df = getattr(computer, group)(df, factor, **spec)
            df.sort_values(by=['id'], inplace=True)
            if conf['factors_pd']['factors'][group][factor].get('factors', None) is None:
                dfc.write_column(factor, df[factor].values)
            else:
                for fout in conf['factors_pd']['factors'][group][factor].get('factors'):
                    fname = factor + '_' + fout
                    dfc.write_column(fname, df[fname].values)
            del df
            gc.collect()

if __name__ == '__main__':
    main(project().conf)
