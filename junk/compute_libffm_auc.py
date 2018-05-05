import pickle
from os.path import abspath, join as join_path
from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from lib.columns import DataFrameCols


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dump', type=str)
    parser.add_argument('fold', type=int)
    args = parser.parse_args()

    dumpdir = abspath(args.dump)
    datadir = abspath('../data/columns')

    dfc = DataFrameCols(datadir)
    df = dfc.load_df(columns=['id', 'is_attributed'])
    df = df[df['is_attributed'] >= 0]

    with open(join_path(dumpdir, 'folds.pkl'), 'rb') as f:
        folds = pickle.load(f)

    train_pred_file = join_path(dumpdir, 'train_pred_%d.txt' % args.fold)
    with open(train_pred_file, 'r') as f:
        p_train = np.array([float(s) for s in f.readlines()])

    valid_pred_file = join_path(dumpdir, 'valid_pred_%d.txt' % args.fold)
    with open(valid_pred_file, 'r') as f:
        p_valid = np.array([float(s) for s in f.readlines()])

    fold_idx = folds[args.fold][0]
    valid_idx = folds[args.fold][1]

    y_train = df.loc[fold_idx, 'is_attributed'].values
    y_valid = df.loc[valid_idx, 'is_attributed'].values

    print('Train results: log_loss=%f, auc=%f' % (log_loss(y_train, p_train), roc_auc_score(y_train, p_train)))
    print('Valid results: log_loss=%f, auc=%f' % (log_loss(y_valid, p_valid), roc_auc_score(y_valid, p_valid)))



# ffm-train -p valid_fold_0.txt -l 0.0002 -k 4 -t 2 train_fold_0.txt model_fold_0.bin
# Train results: log_loss=0.007449, auc=0.964776
# Valid results: log_loss=0.007866, auc=0.961628


