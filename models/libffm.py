import gc
import logging
import subprocess
import pickle
import csv

from os import chdir
from os.path import abspath, join as join_path

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lib.project import project
from lib.columns import DataFrameCols
from lib.utils import makedirs


def write_libffm_data(df, target, fields, shifts):
    df['data'] = df[target].astype(str)
    for k, v in shifts.items():
        print(k)
        df['data'] += ' %d:' % fields[k]
        df['data'] += (df[k] + v).astype(str)
        df['data'] += ':1'
        df.drop(columns=[k], inplace=True)
        gc.collect()
    return df


def main(conf):
    dump_dir = abspath(conf['libffm']['dump']['dir'])
    makedirs(dump_dir)

    data_dir = abspath(conf['libffm']['data']['dir'])
    dfc = DataFrameCols(data_dir)

    target = 'is_attributed'
    fields = {'ip': 0, 'app': 1, 'device': 2, 'os': 3, 'channel': 4}
    shifts = {'ip': 0, 'app': 364779, 'device': 365548, 'os': 369776, 'channel': 370733}

    # 1) write test data
    # logging.info('Writing test data in libffm format')
    # df = dfc.load_df(columns=['id', target] + list(fields.keys()))
    # df = df[df[target] == -1]
    # df[target] = 0  # do we need this?
    # df = write_libffm_data(df, target, fields, shifts)
    test_fname = join_path(dump_dir, 'test.txt')
    # df[['data']].to_csv(test_fname, header=False, index=False, quoting=csv.QUOTE_NONE)
    # del df
    # gc.collect()
    # exit()

    # 2) write training folds
    # logging.info('Writing k-fold training data')
    # df = dfc.load_df(columns=['id', target] + list(fields.keys()))
    # df = df[df[target] >= 0]
    # df = write_libffm_data(df, target, fields, shifts)
    #
    # folds = []
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    # for fold_idx, valid_idx in skf.split(df['id'].values, df[target].values):
    #     folds.append((fold_idx, valid_idx))
    #
    # with open(join_path(dump_dir, 'folds.pkl'), 'wb') as f:
    #     pickle.dump(folds, f)
    #
    # for j_fold, (fold_idx, valid_idx) in enumerate(folds):
    #     logging.info('Writing fold %d in libffm format', j_fold)
    #     train_fname = join_path(dump_dir, 'train_fold_%d.txt' % j_fold)
    #     df.loc[fold_idx, ['data']].to_csv(train_fname, header=False, index=False, quoting=csv.QUOTE_NONE)
    #     valid_fname = join_path(dump_dir, 'valid_fold_%d.txt' % j_fold)
    #     df.loc[valid_idx, ['data']].to_csv(valid_fname, header=False, index=False, quoting=csv.QUOTE_NONE)
    #
    # del df
    # gc.collect()
    # exit()

    df = dfc.load_df(columns=['id', target])
    df = df[df[target] >= 0]

    with open(join_path(dump_dir, 'folds.pkl'), 'rb') as f:
        folds = pickle.load(f)

    chdir(dump_dir)
    for j_fold, (fold_idx, valid_idx) in enumerate(folds):
        logging.info('Training on fold %d', j_fold)
        train_fname = join_path(dump_dir, 'train_fold_%d.txt' % j_fold)
        valid_fname = join_path(dump_dir, 'valid_fold_%d.txt' % j_fold)
        model_fname = join_path(dump_dir, 'model_%d.bin' % j_fold)
        proc = subprocess.run([
            'ffm-train',
            '-p', valid_fname,
            '-l', str(conf['libffm']['options']['lambda']),
            '-k', str(conf['libffm']['options']['factor']),
            '-r', str(conf['libffm']['options']['learning_rate']),
            '-t', str(conf['libffm']['options']['num_iter']),
            train_fname,
            model_fname
        ], stdout=subprocess.PIPE, check=True)

        logging.info('Running command %s', ' '.join(proc.args))
        logging.info('Process return code %d', proc.returncode)
        logging.info(proc.stdout.decode('utf-8'))

        train_pred_file = join_path(dump_dir, 'train_pred_%d.txt' % j_fold)
        proc = subprocess.run([
            'ffm-predict',
            train_fname,
            model_fname,
            train_pred_file
        ], stdout=subprocess.PIPE, check=True)

        logging.info('Running command %s', ' '.join(proc.args))
        logging.info('Process return code %d', proc.returncode)

        with open(train_pred_file, 'r') as f:
            p_train = np.array([float(s) for s in f.readlines()], dtype=np.float32)
            auc_train = roc_auc_score(df.loc[fold_idx, target].values, p_train)

        valid_pred_file = join_path(dump_dir, 'valid_pred_%d.txt' % j_fold)
        proc = subprocess.run([
            'ffm-predict',
            valid_fname,
            model_fname,
            valid_pred_file
        ], stdout=subprocess.PIPE, check=True)

        logging.info('Running command %s', ' '.join(proc.args))
        logging.info('Process return code %d', proc.returncode)

        with open(valid_pred_file, 'r') as f:
            p_valid = np.array([float(s) for s in f.readlines()], dtype=np.float32)
            auc_valid = roc_auc_score(df.loc[valid_idx, target].values, p_valid)

        logging.info('Fold quality: auc_train=%f auc_valid=%f', auc_train, auc_valid)

        test_pred_file = join_path(dump_dir, 'test_pred_%d.txt' % j_fold)
        proc = subprocess.run([
            'ffm-predict',
            test_fname,
            model_fname,
            test_pred_file
        ], stdout=subprocess.PIPE, check=True)

        logging.info('Running command %s', ' '.join(proc.args))
        logging.info('Process return code %d', proc.returncode)


if __name__ == '__main__':
    main(project().conf)
