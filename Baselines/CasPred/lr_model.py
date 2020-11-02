import numpy as np
import json
import os
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from pickle import dump


def generate_invalid():
    """Those subreddits that have 0 >=10 comments in Oct or Nov Dataset are invlaid for our usecase."""
    invalid = set()

    def get_month_invalid(month):
        unknown = "unknown_counts_" + month + ".json"
        known = "known_counts_" + month + ".json"
        with open(os.path.join(main_dir, unknown), "r") as f:
            unknown = json.load(f)
        with open(os.path.join(main_dir, known), "r") as f:
            known = json.load(f)
        for k1, k2 in zip(known, unknown):
            if known[k1] + unknown[k2] == 0:
                invalid.add(int(k1))

    for month in ['oct', 'nov']:
        get_month_invalid(month)
    return invalid


def get_df(month, invalid):
    """Load the feature_month.csv."""
    input_file = "features1_" + month + ".csv"
    col_names = [
        "sub_reddit_id", "post_utc", "text_complexity", "text_readability",
        "text_num_sen", "text_num_word", "text_informative", "text_polarity",
        "text_count_url", "temporal_1", "temporal_2", "temporal_3", "2K", "5K",
        "10K", "20K", "50K", "100K"
    ]
    df = pd.read_csv(os.path.join(main_dir, input_file),
                     header=None,
                     names=col_names)
    for i in invalid:
        df = df[df.sub_reddit_id != i]
    return df


def get_input_features(features_all=True):
    """The training features, either all or selected."""
    if features_all:
        X = df[[
            "post_utc",
            "text_complexity",
            "text_readability",
            "text_num_sen",
            "text_num_word",
            "text_informative",
            "text_polarity",
            "text_count_url",
            "temporal_1",
            "temporal_2",
            "temporal_3",
        ]]
        # One-hot key encoding for subreddit_id.
        S = pd.get_dummies(df[['sub_reddit_id']],
                           prefix_sep="__",
                           columns=['sub_reddit_id'])
        S = np.array(S)
        X = np.concatenate((S, X), axis=1)
        print("S: ", S.shape)
    else:
        X = df[["text_polarity", "temporal_1", "temporal_2", "temporal_3"]]
    X = np.array(X)
    X = normalize(X)
    print("X: ", X.shape)
    return X


def get_output_features():
    """Our label set consists of where the cascade will reach x*k or not."""
    Y_labels = df[keys]
    Y_labels = np.array(Y_labels).astype(float)
    print("Y: ", Y_labels.shape)
    return Y_labels


def train_models(X, Y_labels, keys):
    """for each x*k type train a LR model."""
    """TODO: implement cross-validation."""
    for i in range(6):
        y = Y_labels[:, i]
        key = keys[i]
        clf = LogisticRegression(solver='lbfgs', class_weight='balanced')
        clf.fit(X, y)
        model_dict[key] = clf


def score_models(X, Y, keys):
    """In addition to model.score(), return the kendalltau and spearman rank values as well for each model."""
    y_pred = []
    y_ground = []
    for i in range(6):
        y = Y[:, i]
        y_ground.extend(list(y))
        k = keys[i]
        model_score_dict[k] = {}
        clf = model_dict[k]
        y_p = clf.predict(X)
        y_pred.extend(list(y_p))
        tau, _ = stats.kendalltau(y, y_p)
        spr, _ = stats.spearmanr(y, y_p)
        model_score_dict[k]["score"] = clf.score(X, y)
        model_score_dict[k]["tau"] = tau
        model_score_dict[k]["spr"] = spr
    n_ground = np.array(y_ground)
    n_pred = np.array(y_pred)
    tau, _ = stats.kendalltau(n_ground, n_pred)
    spr, _ = stats.spearmanr(n_ground, n_pred)
    model_score_dict["overall_tau"] = tau
    model_score_dict["overall_spr"] = spr


keys = ["2K", "5K", "10K", "20K", "50K", "100K"]
main_dir = os.path.join("data", "reddit_data")
invalid = generate_invalid()

for i in list([True, False]):
    print("################")
    if i:
        print("Full Features")
    else:
        print("Selected Features")
    model_dict = {}
    model_score_dict = {}
    df = get_df('oct', invalid)
    print("Load Train data")
    X_train = get_input_features(True)
    Y_train_labels = get_output_features()
    print("Train")
    train_models(X_train, Y_train_labels, keys)
    df = get_df('nov', invalid)
    print("Load Test data")
    X_test = get_input_features(True)
    Y_test_labels = get_output_features()
    print("Evaluate")
    score_models(X_test, Y_test_labels, keys)
    if i:
        model_path = os.path.join(main_dir, 'CasPred_full.pkl')
        model_score_path = os.path.join(main_dir,
                                        'CasPred_evaluation_metric_full.json')
    else:
        model_path = os.path.join(main_dir, 'CasPred_org.pkl')
        model_score_path = os.path.join(main_dir,
                                        'CasPred_evaluation_metric_org.json')
    dump(model_dict, open(model_path, 'wb'), protocol=2)
    with open(model_score_path, 'w') as f:
        json.dump(model_score_dict, f, indent=True)
    print("Save Results")
    print("################")
