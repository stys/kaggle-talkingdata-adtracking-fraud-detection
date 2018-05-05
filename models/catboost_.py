# -*- coding: utf-8 -*-

import logging
import json
from os import chdir, getcwd
from os.path import join as join_path, abspath
from copy import deepcopy

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe

from catboost import CatBoostClassifier

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


def train_catboost(train_df, valid_df, target, features, categorical_features, options):
    logging.info('Training catboost with options: %s', options)

    cat_features = list(train_df[features].columns.get_loc(c) for c in categorical_features)

    model = CatBoostClassifier(**options)
    model.fit(X=train_df[features].values, y=train_df[target].values, cat_features=cat_features,
              eval_set=(valid_df[features].values, valid_df[target].values))

    model.save_model('model.bin')

    train_quality = quality(train_df[target].values, model.predict_proba(train_df[features].values)[:, 1])
    logging.info('Train quality: %s', train_quality)

    valid_quality = quality(valid_df[target].values, model.predict_proba(valid_df[features].values)[:, 1])
    logging.info('Validation quality: %s', valid_quality)

    return train_quality, valid_quality, model


def get_hyperopt_objective(train_df, valid_df, target, features, categorical_features, catboost_options):
    """ Construct hyperopt objective function """
    hyperopt_trial = 0

    def hyperobj(params):
        nonlocal hyperopt_trial
        hyperopt_trial += 1
        logging.info('Hyperopt trial %d, params=%s' % (hyperopt_trial, params))

        options = deepcopy(catboost_options)
        for p in params:
            options[p] = params[p]

        work_dir = getcwd()
        trial_dir = abspath(join_path(work_dir, 'trial_%d' % hyperopt_trial))
        makedirs(trial_dir)
        chdir(trial_dir)
        logging.info('Trial directory: %s', trial_dir)

        logging.info('Train catboost with options: %s' % config2json(options))
        train_quality, valid_quality, model = train_catboost(
            train_df, valid_df, target, features, categorical_features, options)

        model.save('model')
        chdir(work_dir)

        return {
            'loss': 1.0 - valid_quality['auc'],
            'status': STATUS_OK,
            'options': config2json(options),
            'quality': {
                'train': train_quality,
                'valid': valid_quality
            },
            'model': {
                'file': join_path(trial_dir, 'model')
            }
        }

    return hyperobj


def train_catboost_with_hyperopt(train_df, valid_df, target, features, categorical_features, catboost_options, hyperopt_options):
    logging.info('Running hyper parameters optimization: %s', config2json(hyperopt_options))

    space = dict()
    for param, opts in hyperopt_options['space'].items():
        expression = getattr(hp, opts['expression'])
        space[param] = expression(label=param, **opts['params'])

    fcn = get_hyperopt_objective(train_df, valid_df, target, features, categorical_features, catboost_options)

    trials = Trials()
    opt = fmin(
        fn=fcn,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=hyperopt_options['max_evals']
    )

    with open('hyperopt_trials.json', 'w') as f:
        json.dump(trials.results, f, indent=4)

    logging.info('Best parameters: %s', opt)

    best_trial, best_trial_result = min(enumerate(trials.results), key=lambda r: r[1]['loss'])
    logging.info('Best model %d: AUC=%s, model=%s' % (
        best_trial, best_trial_result['quality']['valid']['auc'], best_trial_result['model']['file']))

    best_model = CatBoostClassifier()
    best_model.load_model(best_trial_result['model']['file'])
    return best_trial_result['quality']['train'], best_trial_result['quality']['valid'], best_model


if __name__ == '__main__':
    conf = project().conf

    dump_dir = abspath(conf['catboost']['dump']['dir'])
    makedirs(dump_dir)

    write_config(conf, join_path(dump_dir, 'application.conf'), 'hocon')
    write_config(conf, join_path(dump_dir, 'application.json'), 'json')
    logging.getLogger().addHandler(logging.FileHandler(join_path(dump_dir, 'application.log')))

    logging.info('Kaggle Talking Data')
    logging.info('Train Catboost')
    logging.info('Dump: %s', dump_dir)

    target = conf['catboost']['target']
    features = conf['catboost']['features']
    categorical_features = conf['catboost']['categorical_features']
    logging.info('Target: %s', target)
    logging.info('Features: %s', config2json(features))
    logging.info('Categorical features: %s', categorical_features)

    data_dir = abspath(conf['catboost']['data']['dir'])
    dfc = DataFrameCols(data_dir)

    train_index_name = conf['catboost']['data']['train']['index']
    train_index = dfc.load_index(train_index_name)
    train_df = dfc.load_df(columns=[target] + features, index=train_index)
    train_df, valid_df = train_test_split(train_df, test_size=0.1)

    catboost_options = conf['catboost']['options']
    logging.info('Using catboost options: %s', catboost_options)

    work_dir = getcwd()
    chdir(dump_dir)

    hyperopt_options = conf['catboost']['hyperopt']
    if hyperopt_options['enabled']:
        train_quality, valid_quality, model = train_catboost_with_hyperopt(train_df, valid_df, target, features, categorical_features, catboost_options, hyperopt_options)
    else:
        train_quality, valid_quality, model = train_catboost(train_df, valid_df, target, features, categorical_features, catboost_options)

    chdir(work_dir)

    valid_pred = model.predict_proba(valid_df[features].values)[:, 1]
    valid_quality = quality(valid_df[target].values, valid_pred)
    logging.info('Cross-check best model validation score: AUC=%s' % valid_quality['auc'])

    # model = CatBoostClassifier()
    # model.load_model(join_path(dump_dir, 'model.bin'))

    test_index_name = conf['catboost']['data']['test']['index']
    test_index = dfc.load_index(test_index_name)
    test_df = dfc.load_df(columns=features + ['click_id_submission'], index=test_index)
    test_df['is_attributed'] = model.predict_proba(test_df[features].values)[:, 1]
    test_df = test_df[['click_id_submission', 'is_attributed']].rename(columns={'click_id_submission': 'click_id'})
    test_df.sort_values(by='click_id', inplace=True)
    test_df.to_csv(join_path(dump_dir, 'submission.csv'), header=True, index=False)
