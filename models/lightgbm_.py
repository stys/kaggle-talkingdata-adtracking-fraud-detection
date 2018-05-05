import gc
import logging
from os.path import abspath, join as join_path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

import lightgbm as lgb

from lib.project import project
from lib.columns import DataFrameCols
from lib.utils import makedirs
from lib.hocon import write_config, config2json
from lib.quality import reliability_curve


def quality(labels, pred):
    return dict(
        ll=log_loss(labels, pred),
        auc=roc_auc_score(labels, pred),
        reliability=list(map(lambda x: x.tolist(), reliability_curve(labels, pred, nbins=100)))
    )


def train_lightgbm(params, train_dataset, valid_dataset=None, **options):
    logging.info('Training LightGBM with params: %s', config2json(params))
    model = lgb.train(params, train_dataset, valid_sets=[train_dataset, valid_dataset], **options)
    return model


def main(conf):
    dump_dir = conf['lightgbm']['dump']['dir']
    makedirs(dump_dir)

    write_config(conf, join_path(dump_dir, 'application.conf'), 'hocon')
    write_config(conf, join_path(dump_dir, 'application.json'), 'json')
    logging.getLogger().addHandler(logging.FileHandler(join_path(dump_dir, 'application.log')))

    logging.info('Kaggle Talking Data')

    label = conf['lightgbm']['label']
    features = conf['lightgbm']['features']
    categorical_features = conf['lightgbm']['categorical_features']
    logging.info('Label: %s', label)
    logging.info('Features: %s', features)
    logging.info('Categorical features: %s', categorical_features)

    data_dir = abspath(conf['lightgbm']['data']['dir'])
    dfc = DataFrameCols(data_dir)
    train_index_name = conf['lightgbm']['data']['train']['index']
    train_index = dfc.load_index(train_index_name)

    df = dfc.load_df(columns=[label] + features, index=train_index)

    if conf['lightgbm']['valid_size'] > 0:
        train_df, valid_df = train_test_split(df, test_size=conf['lightgbm']['valid_size'])

        train_dataset = lgb.Dataset(data=train_df[features].values, label=train_df[label].values, feature_name=features,
                                    categorical_feature=categorical_features)
        valid_dataset = lgb.Dataset(data=valid_df[features].values, label=valid_df[label].values, feature_name=features,
                                    categorical_feature=categorical_features)

        del train_df
        del valid_df
        gc.collect()
    else:
        train_dataset = lgb.Dataset(data=df[features].values, label=df[label].values, feature_name=features,
                                    categorical_feature=categorical_features)
        valid_dataset = None

    params = conf['lightgbm']['params']
    options = conf['lightgbm']['options']
    model = train_lightgbm(params, train_dataset, valid_dataset, **options)
    model.save_model(join_path(dump_dir, 'model.bin'))
    del train_dataset
    del valid_dataset
    gc.collect()

    # train_label = train_df[label].values
    # train_pred = model.predict(train_df[features])
    # train_quality = quality(train_label, train_pred)
    # logging.info('Train quality: %s', train_quality)
    #
    # valid_label = valid_df[label].values
    # valid_pred = model.predict(valid_df[features])
    # valid_quality = quality(valid_label, valid_pred)
    # logging.info('Valid quality: %s', valid_quality)

    test_index_name = conf['lightgbm']['data']['test']['index']
    test_index = dfc.load_index(test_index_name)
    test_df = dfc.load_df(columns=features + ['click_id_submission'], index=test_index)
    test_df['is_attributed'] = model.predict(test_df[features])
    test_df = test_df[['click_id_submission', 'is_attributed']].rename(columns={'click_id_submission': 'click_id'})
    test_df.sort_values(by='click_id', inplace=True)
    test_df.to_csv(join_path(dump_dir, 'submission.csv'), header=True, index=False)

    gain = model.feature_importance('gain')
    ft = pd.DataFrame({
        'feature': model.feature_name(),
        'split': model.feature_importance('split'),
        'gain': 100 * gain / gain.sum()}
    ).sort_values('gain', ascending=False)
    ft.to_csv(join_path(dump_dir, 'feature_strength.csv'), header=True, index=False, sep='\t')

if __name__ == '__main__':
    main(project().conf)
