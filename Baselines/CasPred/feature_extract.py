import json
import csv
import pickle
import re
import os, sys
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize as st
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from senticnet.senticnet import SenticNet
import argparse


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


parser = argparse.ArgumentParser(description='Input Params')
parser.add_argument('--m', help='month')
args = parser.parse_args()
month = args.m if args.m else 'oct'

stop_words = list(stopwords.words('english'))

K = 10
y_label_set = [2 * K, 5 * K, 10 * K, 20 * K, 50 * K, 100 * K]

main_dir = os.path.join("data", "reddit_data")
input_file = "selected_discussion_" + month + ".jsonlist"

with open(os.path.join(main_dir, input_file), "r") as f:
    data = f.readlines()
n = len(data)
print(n)

tf_path = os.path.join(main_dir, 'tfidf_transformer_' + month + '.pkl')
with open(tf_path, 'rb') as f:
    tfidf_transformer = pickle.load(f)
cv_path = os.path.join(main_dir, 'count_vectorizer_' + month + '.pkl')
with open(cv_path, 'rb') as f:
    cv = pickle.load(f)
feature_names = cv.get_feature_names()

sn = SenticNet()
sn_words = sn.__dict__['data'].keys()
unknown_dict = {}
known_dict = {}
subreddit_index_map = {}


def complexity(c):
    """Score how complex the post is."""
    idx = np.nonzero(c)[0]
    if len(idx) == 0:
        return 0.0
    s = 0
    for i in idx:
        s += c[i] * (np.log(len(idx)) - np.log(1 + c[i]))
    return s / len(idx)


def num_sen(text):
    """How long is the post."""
    return len(st(text))


def num_word(text, c):
    """Number of word tokens in the post."""
    ns = num_sen(text)
    if ns:
        return np.sum(c) / num_sen(text)
    else:
        return 0.0


def readability(text, c):
    """Score the post on readability."""
    idx = np.nonzero(c)[0]
    if len(idx) == 0:
        return 0.0
    h_word = 0
    for i in idx:
        if len(feature_names[i]) > 6:
            h_word += 1
    return 100 * float(h_word) / np.sum(c) + num_word(text, c)


def informative(text):
    """How informative is the the post."""
    tfidf_vector = tfidf_transformer.transform(cv.transform([text
                                                             ])).toarray()[0]
    return np.sum(tfidf_vector)


def polarity(c, sub_red):
    """What is the sentiment expressed by the post."""
    idx = np.nonzero(c)[0]
    score = 0
    for i in idx:
        if feature_names[i] not in sn_words:
            unknown_dict[sub_red] += 1
        else:
            known_dict[sub_red] += 1
            score += float(sn.polarity_intense(feature_names[i])) * c[i]
    return score


def count_url(s):
    """Number of external links in the post."""
    return len(re.findall(r"http\S+", s))


def temporal_1(thread_create_utc, thread_comments_utc, k):
    """Return the time between post and kth comment."""
    return thread_comments_utc[k - 1] - thread_create_utc


def temporal_2(thread_comments_utc, k):
    """Return the avg reshare time diff for first k/2 posts."""
    k = k // 2
    reshare_diff = 0
    for i in range(1, k):
        reshare_diff += thread_comments_utc[i] - thread_comments_utc[i - 1]
    return reshare_diff / (k - 1)


def temporal_3(thread_comments_utc, k):
    """Return the avg reshare time diff for last k/2 posts."""
    k = k // 2
    reshare_diff = 0
    for i in range(1, k):
        reshare_diff += thread_comments_utc[-i] - thread_comments_utc[-i - 1]
    return reshare_diff / (k - 1)


def y_label_features(thread_comments_utc, k):
    """If total length will reach x*k size or not."""
    l = len(thread_comments_utc)
    return [l >= yl for yl in y_label_set]


feature_list = []

for i in range(n):
    print("################")
    print(i)
    unknown_dict[i] = 0
    known_dict[i] = 0
    each_reddit = json.loads(data[i])
    value = list(each_reddit.keys())[0]
    subreddit_index_map[i] = value
    each_reddit = each_reddit[value]
    for thread in each_reddit:
        if len(thread['comments']) < 10:
            continue
        thread_feature = []
        raw_text = thread['selftext'] + '\n' + thread['title']
        raw_text = re.sub(r'\d+|[^\w\s]', '', raw_text)
        cleaned_text = re.sub(r"http\S+", '', raw_text).lower()
        cvector = cv.transform([cleaned_text]).toarray()[0]
        thread_feature.append(i)
        thread_create_utc = thread['created_utc']
        thread_feature.extend([
            thread_create_utc,
            complexity(cvector),
            readability(cleaned_text, cvector),
            num_sen(cleaned_text),
            num_word(cleaned_text, cvector),
            informative(cleaned_text),
            polarity(cvector, i),
            count_url(raw_text)
        ])
        thread_comments_utc = [
            comment['created_utc'] for comment in thread['comments']
        ]
        thread_feature.extend([
            temporal_1(thread_create_utc, thread_comments_utc, K),
            temporal_2(thread_comments_utc, K),
            temporal_3(thread_comments_utc, K)
        ])
        thread_feature.extend(y_label_features(thread_comments_utc, K))
        feature_list.append(thread_feature)

feature_filename = "features1_" + month + ".csv"
with open(os.path.join(main_dir, feature_filename), "w") as f:
    wr = csv.writer(f)
    wr.writerows(feature_list)
# known-unknown counts: Number of words in post known-unknown by the sn model.

unknown_counts_filename = "unknown_counts_" + month + ".json"
with open(os.path.join(main_dir, unknown_counts_filename), "w") as f:
    json.dump(unknown_dict, f, indent=True)
known_counts_filename = "known_counts_" + month + ".json"
with open(os.path.join(main_dir, known_counts_filename), "w") as f:
    json.dump(known_dict, f, indent=True)

# map subreddit_name to its id.
subreddit_map_name = "subreddit_index_map_" + month + ".json"
with open(os.path.join(main_dir, subreddit_map_name), "w") as f:
    json.dump(subreddit_index_map, f, indent=True)
