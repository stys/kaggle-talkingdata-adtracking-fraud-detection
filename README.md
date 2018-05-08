Kaggle TalkingData Ad Tracking Fraud Detection Challenge

Scripts for competition
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

Final ranking #55 with private leaderboard score 0.9824250

* Aggregate factors from Baris' kernel computed on all data with test_supplement
* Prev/next click times (1-, 2-, 3-step) computed on all data with test_supplement
* LIBFFM out-of-fold using 5 folds
* LGBM on 50% data (using all is_attributed=1 and 50% of is_attributed=0)
* Average of a few LGBM models trained on different subsets of data and factors
* Training with 96GB RAM
* Store data column-wise in binary formats for fast loading
* HOCON configurations are awesome

EDA [preprocessing/eda.ipyndb](preprocessing/eda.ipynb) - distributions of frequencies in train.csv

Merge test datasets [preprocessing/merge_test_sets.ipynb](preprocessing/merge_test_sets.ipynb) - script for merging test datasets
test.csv
test_supplement.csv

Baris kernel
https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977?scriptVersionId=3224614
https://www.kaggle.com/aharless/kaggle-runnable-version-of-baris-kanber-s-lightgbm

Next/prev clicks without hashing trick
https://www.kaggle.com/asydorchuk/nextclick-calculation-without-hashing-trick

Submissions

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
blend    |            |                    | logit, weights=1.0                | -       | 0.9812   | blend lgbm_13..lgbm_17
lgbm     | lgbm_18    | na_20pct           | it=1500 lr=0.1 md=3 nl=7 scw=na   | 0.98571 | 0.9808   | baris + t2 + libffm + tc2
lgbm     | lgbm_19    | na_50pct           | it=2500 lr=0.1 md=3 nl=7 scw=na   | 0.98595 | 0.9811   | baris + t2
blend    |            |                    | logit, weights=1.0                | -       | 0.9813   | blend lgbm_15 + lgbm_19
lgbm     | lgbm_19    | na_50pct           | it=1500, all attributed           | -       | 0.9810   | baris + t2 + libffm
lgbm     | lgbm_20    | na_50pct_2         | it=1500, all attributed           | -       | 0.9810   | baris + t2 + libffm
blend    |            |                    | logit, weights=1.0                | -       | 0.9813   | blend lgbm_13..lgbm_20

Factors strength (lgbm_19)

feature | gain | split
--- | --- | ---
libffm_oof | 82.34410715832054 | 363
ip_app_device_os_t_next | 4.858614532478393 | 555
app | 4.285458537390102 | 1057
ip_nunique_channel | 1.7134587347643337 | 173
channel | 1.2185210564277669 | 2220
os | 0.9917807676129159 | 1516
hour | 0.8615137822058627 | 1078
ip_nunique_app | 0.5879513478431967 | 157
ip_nunique_device | 0.5397789939897564 | 190
ip_day_hour_count | 0.461685682523349 | 191
ip_app_count | 0.42650563120670515 | 99
ip_app_device_os_t_next_2 | 0.35302018803960517 | 194
ip_device_os_nunique_app | 0.3238393490783555 | 198
ip_day_nunique_hour | 0.30262413717998854 | 76
ip_app_os_count | 0.2384258533635993 | 84
ip_device_os_cumcount_app | 0.11690381200637338 | 58
ip_app_device_os_t_prev | 0.07133512725949863 | 95
device | 0.06228725573269818 | 70
ip_app_nunique_os | 0.06172545078968752 | 141
ip_cumcount_os | 0.05417631528656273 | 44
ip_app_channel_mean_hour | 0.04849352774733603 | 144
day | 0.017391508495188064 | 61
app_nunique_channel | 0.014128035383901352 | 35
ip_app_os_var_hour | 0.01325167381501645 | 52
ip_app_device_os_t_prev_2 | 0.012127423364797763 | 60
ip_app_channel_var_day | 0.011027400971738753 | 39
ip_day_channel_var_hour | 0.009866716722734953 | 50
