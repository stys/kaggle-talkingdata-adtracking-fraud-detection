import logging
import pickle
from os.path import abspath, join as join_path
from argparse import ArgumentParser

import numpy as np

from scipy.special import logit
from sklearn.metrics import roc_auc_score, log_loss

from lib.columns import DataFrameCols


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dump')
    args = parser.parse_args()

    dumpdir = abspath(args.dump)
    datadir = abspath('../data/columns')

    dfc = DataFrameCols(datadir)
    df = dfc.load_df(columns=['id', 'is_attributed'])
    df['p'] = 0

    df_train = df[df['is_attributed'] >= 0]
    df_test = df[df['is_attributed'] == -1]
    print(df_test.shape[0])

    with open(join_path(dumpdir, 'folds.pkl'), 'rb') as f:
        folds = pickle.load(f)

    p_test_avg = np.zeros(df_test.shape[0])
    for j_fold, (fold_idx, valid_idx) in enumerate(folds):
        valid_pred_file = join_path(dumpdir, 'valid_pred_%d.txt' % j_fold)
        with open(valid_pred_file, 'r') as f:
            p_valid = np.array([float(s) for s in f.readlines()])

        y_valid = df_train.loc[valid_idx, 'is_attributed'].values
        auc_valid = roc_auc_score(y_valid, p_valid)
        print('Fold %d validation auc=%f' % (j_fold, auc_valid))

        df_train.loc[valid_idx, 'p'] = logit(p_valid)

        test_pred_file = join_path(dumpdir, 'test_pred_%d.txt' % j_fold)
        with open(test_pred_file, 'r') as f:
            p_test = np.array([float(s) for s in f.readlines()])
            p_test_avg += logit(p_test)

    df_test.loc[:, 'p'] = p_test_avg / 5
    df_all = df_train.append(df_test, ignore_index=True)
    df_all.sort_values(by=['id'], inplace=True)
    dfc.write_column('libffm_oof', df_all['p'].values)


