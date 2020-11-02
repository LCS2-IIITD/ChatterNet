max_sub_tokens = 100
max_news_tokens = 200
max_sub_per_min = 50
max_news_per_min = 100
from datetime import datetime
import sys
import os
import pickle
import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.tokenize import word_tokenize
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from num2word import num2word
def my_replace(match):
    match = match.group()
    try:
        return ' '+num2word(round(float(match)))+' '
    except Exception:
        return ' '
lemmatizer = WordNetLemmatizer()



news_articles  = json.load(open('../News_articles/source_news_ordered.json','r'))
source_count = json.load(open('../News_articles/source_article_count.json','r'))
comment_rate = json.load(open('../Data_processing/comment_rate_november.json','r'))
ts_list = json.load(open('../Data_processing/disc_id2ts_nov.json','r'))
word2vec_model = Word2Vec.load('wvmodel.model')

def text_to_array(text, max_len):
    arr = []
    for sent in sent_tokenize(re.sub(r'http\S+|[^\x00-\x7f]','', text)):
        s=re.sub(r'[0-9]+[\.[0-9]+$]*', my_replace, " ".join(re.split('\W+|_', sent)).lower())
        for w in s.split():
            try:
                arr.append(word2vec_model.wv.vocab[w].index+1)
            except KeyError:
                pass
    arr = list(pad_sequences([arr], padding='post', truncating='post', maxlen=max_len, dtype='int32'))
    return arr[0]

def temporal_count(sub_id):
    c_t = []
    for start_ts in range(0, 3600, 60):
        end_ts = start_ts+60
        try:
            count = len([ts for ts in ts_list[sub_id] if (ts<end_ts and ts>start_ts)])
        except KeyError:
            count = 0
        c_t.append(count)
    return c_t
        
def process_subreddit(subs, interval, subred_list):
    start = 1541030400#1538611200#
    end = start + interval
    text_array = []
    time_array = []
    value_array = []
    subred_array = []
    comment_rate_array = []
    comment_ts_array = []
    index = 0
    time_step = 0
    logfile = open('submission_processing_log.txt','w')
    for sub in subs:
        if not index:
            t_arr = []
            ti_arr = []
            v_arr = []
            s_arr = []
            c_r_arr = []
            c_t_arr = []
        if sub['created_utc']>=start:
            if sub['created_utc']<end and index<len(subs)-1:
                t_arr.append(text_to_array(sub['text'], max_sub_tokens))
                ti_arr.append(sub['created_utc'])
                s_arr.append(subred_list.index(sub['subreddit'])+1)
                c_t_arr.append(temporal_count(sub['id']))
                try:
                    c_r_arr.append(comment_rate[sub['subreddit']][time_step])
                except IndexError:
                    c_r_arr.append(0)
                try:
                    v_arr.append(np.log(sub['comment_count']-sum(c_t_arr[-1])+1))
                except KeyError:
                    #print('Comment Not Found')
                    v_arr.append(1)
            else:
                text_array.append(t_arr)
                time_array.append(ti_arr)
                subred_array.append(s_arr)
                value_array.append(v_arr)
                comment_rate_array.append(c_r_arr)
                comment_ts_array.append(c_t_arr)
                logfile.write(str(len(t_arr))+'\t')
                end = end+interval
                time_step+=1
                t_arr = [text_to_array(sub['text'], max_sub_tokens)]
                ti_arr = [sub['created_utc']]
                s_arr = [subred_list.index(sub['subreddit'])+1]
                c_t_arr = [temporal_count(sub['id'])]
                try:
                    c_r_arr = [comment_rate[sub['subreddit']][time_step]]
                except IndexError:
                    c_r_arr = [0]
                try:
                    v_arr = [np.log(sub['comment_count']-sum(c_t_arr[-1])+1)]
                except KeyError:
                    #print('Comment Not Found')
                    v_arr = [1]
                
        index+=1
    logfile.close()
    return text_array, time_array, subred_array, value_array, comment_rate_array, comment_ts_array

def process_news(interval, articles):
    start = 1538611200#1541030400#
    end = start + interval
    text_array = []
    index = 0
    logfile = open('news_processing_log.txt','w')
    for article in articles:
        if not index:
            t_arr = []
        if article['ts']>=start:
            if article['ts']<end and index<len(articles)-1:
                t_arr.append(text_to_array(article['text'], max_news_tokens))
            else:
                text_array.append(t_arr)
                logfile.write(str(len(t_arr))+'\t')
                end = end+interval
                t_arr = [text_to_array(article['text'], max_news_tokens)]
        index+=1
    logfile.close()
    return text_array

def main():
        
    from_subred_file = open('../Reddit dumps/Subreddit_list.txt').read().split('\n')[:-1]
    selected_subred = []
    for s in from_subred_file:
        if s[-1]=='*':
            selected_subred.append(s.split(':')[0])
    #selected_subred = ['Futurology','technology', 'The_Donald','politics', 'science', 'news', 'worldnews']

    with open('../Reddit dumps/selected_submissions_november.json','r') as infile:
        subs = json.load(infile)
    sel_subs = []
    for subred in selected_subred:
        for sub in subs[subred]:
            sub['subreddit'] = subred
            sel_subs.append(sub)

    sorted_subs = sorted(sel_subs, key = lambda i:i['created_utc'])
    text, time, subred, value, crate, ccount = process_subreddit(sorted_subs, 60, selected_subred)
    text = pad_sequences(text, padding='post', truncating='post', maxlen=max_sub_per_min, dtype='int32')
    subred = pad_sequences(subred, padding='post', truncating='post', maxlen=max_sub_per_min, dtype='int32')
    value = pad_sequences(value, padding='post', truncating='post', maxlen=max_sub_per_min, dtype='float64')
    crate = pad_sequences(crate, padding='post', truncating='post', maxlen=max_sub_per_min, dtype='float64')
    ccount = pad_sequences(ccount, padding='post', truncating='post', maxlen=max_sub_per_min, dtype='float64')
    value = np.reshape(value, value.shape+(1,))
    print('Shape of text array:',text.shape)
    print('Shape of subreddit array:',subred.shape)
    print('Shape of value array:', value.shape)
    print('Shape of comment rate array:', crate.shape)
    print('Shape of comment count array:', ccount.shape)
    np.save('submission_value_60min_november.npy', value)
    np.save('submission_text_november.npy', text)
    np.save('submission_subred_november.npy', subred)
    np.save('submission_comment_rate_november.npy', crate)
    np.save('temporal_cc60min_november.npy', ccount)
##    
##    articles = []
##    for source in list(reversed(source_count))[:100]:
##        for article in news_articles[source['source']]['articles']:
##            date_obj = datetime.strptime(article['date_publish'], '%Y-%m-%d %H:%M:%S')
##            ts = round(date_obj.timestamp() + (datetime.now().timestamp() -datetime.utcnow().timestamp()))
##            discard=False
##            d = {}
##            d['ts'] = ts
##            try:
##                d['text'] = article['title']+'\n'+article['text']
##            except TypeError:
##                try:
##                    d['text'] = article['title']+'\n'+article['description']
##                except TypeError:
##                    discard = True
##            if not discard:
##                articles.append(d)
##    sorted_articles = sorted(articles, key = lambda i:i['ts'])
##    n_text = process_news(60, sorted_articles)
##
##    n_text = pad_sequences(n_text, padding='post', truncating='post', maxlen=max_news_per_min, dtype='int32')
##    
##    print('Shape of news text array:',n_text.shape)
##    np.save('news_text.npy', n_text)
    
    

if __name__ == "__main__":
    main()

