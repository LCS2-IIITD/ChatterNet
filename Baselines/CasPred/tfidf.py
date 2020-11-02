import json
import re
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
from nltk.corpus import stopwords
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

# Load stop words.
stop_words = list(stopwords.words('english'))

# Load the discussion thread. JsonList of json objects.
main_dir = os.path.join("data", "reddit_data")
input_file = "selected_discussion_" + month + ".jsonlist"

with open(os.path.join(main_dir, input_file), "r") as f:
    data = f.readlines()
n = len(data)
print(n)

docs = []
for i in range(n):
    print(i)
    each_reddit = json.loads(data[i])
    each_reddit = each_reddit['subreddit']
    for thread in each_reddit:
        s = " "
        if len(thread['comments']) < 10:
            continue
        s = thread['selftext']
        s = s + '\n' + thread['title']
        s = re.sub(r'\d+|[^\w\s]', '', s)
        s = re.sub(r"http\S+", '', s).lower()
        docs.append(s)

# Generate and save tf-idf model.
print('Documents loaded...')
cv = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words)
word_count_vector = cv.fit_transform(docs)
print('Done countvectorize...')
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)
print('Done tfidf...')
tf_path = os.path.join(main_dir, 'tfidf_transformer_' + month + '.pkl')
dump(tfidf_transformer, open(tf_path, 'wb'), protocol=2)
cv_path = os.path.join(main_dir, 'count_vectorizer_' + month + '.pkl')
dump(cv, open(cv_path, 'wb'), protocol=2)
