# Kaggle TalkingData Ad Tracking Fraud Detection Challenge
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

# Setup

#### Env
Create new conda environment and install basic tools. Using python 3.4 for compatibility with YT.
```
$ conda create --name=kaggle-talkingdata python=3.4 numpy pandas scipy matplotlib
```

```
$ conda list | grep -e 'numpy\|pandas\|scipy\|matplotlib'
matplotlib                2.0.2                    py34_2    conda-forge
numpy                     1.13.1          py34_blas_openblas_200  [blas_openblas]  conda-forge
pandas                    0.20.3                   py34_1    conda-forge
scipy                     0.19.1          py34_blas_openblas_202  [blas_openblas]  conda-forge
```

Update pip
```
$ pip install --upgrade pip
```

Additional packages
```
$ conda install nb_conda                # jupyter notebook support
$ pip install pyhocon                   # HOCON connfiguration
```


#### YT packages and config
[YT API](https://wiki.yandex-team.ru/yt/userdoc/api/) •
[YT Python wrapper](https://wiki.yandex-team.ru/yt/userdoc/pythonwrapper) •
[YT PyDOC](http://pydoc.yt.yandex.net/index.html)

```
$ pip install yandex-yt -i https://pypi.yandex-team.ru/simple
$ pip install yandex-yt-yson-bindings -i https://pypi.yandex-team.ru/simple        # fails on macOS
```

Additional config for python 3.4 compatibility
```
yt.config["pickling"]["python_binary"] = "python3.4"
yt.config["pickling"]["ignore_yson_bindings_for_incompatible_platforms"] = False
```



#### Catboost
[Python tutorial](https://github.com/catboost/catboost/blob/master/catboost/tutorials/catboost_python_tutorial.ipynb)

```
$ pip install catboost
```


#### Hyperopt
[Hyperopt](http://hyperopt.github.io/hyperopt/) •
[Tutorial](https://github.com/hyperopt/hyperopt/wiki/FMin)

```
$ pip install hyperopt
```


# EDA
[preprocessing/eda.ipyndb](preprocessing/eda.ipynb) - distributions of frequencies in train.csv

#### Merge test datasets
[preprocessing/merge_test_sets.ipynb](preprocessing/merge_test_sets.ipynb) - script for merging test datasets
test.csv
test_supplement.csv

Data in test.csv is known to be a subset of data in test_supplement.csv.
There are time gaps between train.csv and test.csv, but fortunately test_supplement.csv contains data for the gaps.
It seems useful to merge all three datasets into a single dataset. About 2% of samples in datasets are exact duplicates.
Care is required when joining test.csv and test_supplement.csv, so that number of duplicates is preserved.


# Factors

#### Baris kernel
https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977?scriptVersionId=3224614
https://www.kaggle.com/aharless/kaggle-runnable-version-of-baris-kanber-s-lightgbm

#### Next/prev clicks without hashing trick
https://www.kaggle.com/asydorchuk/nextclick-calculation-without-hashing-trick


# Models
model    | dump       | train_set          | params                            | val AUC | lb AUC   | factors
---      | ---        | ---                | ---                               | ---     | ---      | ---
lgbm     | lgbm_08    | d_8_9_h_4_5_9_...  | it=250 lr=0.2 md=3 nl=7 scw=300   | 0.983xx | 0.9795   | baris
lgbm     | lgbm_09    | na_10pct           | it=500 lr=0.1 md=3 nl=7 scw=na    | 0.98498 | 0.9803   | baris
lgbm     | lgbm_11    | d_8_9_h_4_5_9_...  | it=2000 lr=0.01 md=3 nl=7 scw=300 | 0.98332 | 0.9791   | baris
cbst     | cbst_09    | na_10pct           | it=500 lr=0.05 md=6 rms=0.7       | 0.98221 | 0.9766   | baris
lgbm     | lgbm_12    | na_10pct           | it=500 lr=0.1 md=3 nl=7 scw=na    | 0.98564 | 0.9806   | baris + t2
lgbm     | lgbm_13    | na_10pct           | it=1000 lr=0.1 md=3 nl=7 scw=na   | 0.98610 | 0.9810   | baris + t2
lgbm     | lgbm_14    | na_10pct           | it=1000 lr=0.1 md=3 nl=7 scw=na   | 0.9855x | 0.9810   | baris + t2 + t3
lgbm     | lgbm_15    | na_10pct           | it=1370 lr=0.1 md=3 nl=7 scw=na   | 0.98567 | 0.9811   | baris + t2 + t3
lgbm     | lgbm_16    | na_10pct           | it=1360 lr=0.1 md=3 nl=7 scw=na   | 0.98544 | 0.9809   | baris + t2 + libffm
lgbm     | lgbm_17    | na_10pct           | it=820 lr=0.1 md=4 nl=15 scw=na   | 0.98580 | 0.9810   | baris + t2 + libffm
lgbm     | blend      |                    | logit, weights=0.2                |         | 0.9812   | blend lgbm_13..lgbm_17
lgbm     | lgbm_18    | na_20pct           | it=1500 lr=0.1 md=3 nl=7 scw=na   | 0.98571 | 0.9808   | baris + t2 + libffm + tc2
lgbm     | lgbm_19    | na_50pct           | it=10000 lr=0.1 md=3 nl=7 scw=na  |


# TODO
[+] catboost version of lgbm_09
[+] time to 2-next/2-prev clicks
[+] time to 3-next/3-prev clicks - no improvement
[ ] lightgbm feature importance
[ ] catboost feature importance
[ ] overflow counters
[+] check libffm auc scores
[+] 5 out-of-fold libffm
[ ] train on all clicks without validation set
[ ] train multiple lgbm models and average together