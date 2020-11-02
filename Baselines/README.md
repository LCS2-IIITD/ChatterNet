# Baselines:
* [CasPred](#CasPred)
* [TiDeH](#TiDeH)
* [DeepCas](#DeepCas)
* RGNet

## CasPred

A modified version of the paper [Can Cascades be Predicted](https://arxiv.org/abs/1403.4608).

### To Run CasPred:
* Load the raw data in the format `data/reddit_data/selected_discussion_<month>.jsonlist`
* Run these files in order
```
python3 tfidf.py --m oct
python3 tfidf.py --m nov
python3 feature_extract.py --m oct
python3 feature_extract.py --m nov
python3 lr_model.py
```
* Your evaluation files should have been generated under `data/reddit_data/CasPred_evaluation_metric_full.json` and `data/reddit_data/CasPred_evaluation_metric_org.json`
* Your LR models Dict will be under `data/reddit_data/CasPred_full.pkl` and `data/reddit_data/CasPred_org.pkl`

## TiDeH
[TiDeH: Time-Dependent Hawkes Process for Predicting Retweet Dynamics](https://arxiv.org/abs/1603.09449)

Original Code by Authors: https://github.com/NII-Kobayashi/TiDeH.

We have used the modified fork: https://github.com/sara-02/TiDeH

### To Run TiDeH:
* Load the raw data in the format `data/reddit_data/selected_discussion_<month>.jsonlist`
* `cd data/reddit_data` 
* Create these main folders `mkdir OCT_INPUT OCT_OUTPUT OCT_OUTPUT_HOUR NOV_INPUT NOV_OUTPUT NOV_OUTPUT_HOUR`
* `cd ../..`
* Populate main folders
```
python3 preprocess_reddit_oct.py
python3 preprocess_reddit_nov.py
```
* Update the `sub_reddit_list` with the folder names in `train_reddit_parts.py`
* Run  train_reddit on server(it took 24hrs each for our Oct and Nov dataset)
```
python3 train_reddit.py --m <MONTH= 'oct' or 'nov'>
```

* Once train_reddits has run, we can run the following evaluations:
`evaluation_metric.py` gives us mse, tau, and row for each subreddit as well as overall dataset.
`evaluation_mse_bins.py` gives us mse on per day basis
`evaluation_cascade.py` gives us overall tau and row for `x*k` similar to `CasPred`
```
python3 evaluation_metric.py --m <MONTH= 'oct' or 'nov'>
python3 evaluation_mse_bins.py --m <MONTH= 'oct' or 'nov'>
python3 evaluation_cascade.py --m <MONTH= 'oct' or 'nov'>
```
* To get untrained estimates of `p(t)` and the per hour window mse
```
python3 cal_hour_counts.py --m <MONTH= 'oct' or 'nov'>
python3 evaluation_rate_metric.py --m <MONTH= 'oct' or 'nov'>
```

## DeepCas
[DeepCas](https://arxiv.org/abs/1611.05373)
Original Code by Authors: https://github.com/chengli-um/DeepCas

Modified fork used by us:https://github.com/sara-02/DeepCas

## Tensorflow Implementation
### Prerequisites
Python 2
Tensorflow 0.12.1

### Parameters used
For `t`, we used  period of `1 hour` window. And for `Delta_t` we used a window of `1 day, 10 days, 30 days.`

### Basic Usage
To run *DeepCas* tensorflow version on a test data set, execute the following command:<br/>
```{r, engine='bash', count_lines}
cd DeepCas
python2 gen_walks/gen_walks.py --dataset <path2dataset>
cd tensorflow
python2 preprocess.py
python2 run.py
```
To run evalutions
```
cd tensorflow
python2 evaluation_metrics.py
python2 evaluation_cascade.py
```
