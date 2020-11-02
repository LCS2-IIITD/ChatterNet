max_token = 150
max_comments = 500
oov_count = 0
import pickle
import json
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.tokenize import word_tokenize
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

def load_pred_time(fname):
    with open(fname,'r') as infile:
        pts = json.load(infile)
    return pts
    
def load_user_embedding(fname):
    with open(fname,'r') as infile:
        ue = json.load(infile)
    return ue
    
def load_disc(fname):
    with open(fname,'r') as infile:
        disc = json.load(infile)
    return disc

def load_vocab(vocab_fname):
    with open(vocab_fname,'r') as infile:
        vocab = json.load(infile)
    return vocab

def text_to_sequence(st, vocab):
    global oov_count
    lwords = []
    text = word_tokenize(re.sub(r'http\S+', '', st))
    for w in text:
        try:
            lwords.append(vocab[lemmatizer.lemmatize(w.lower())])
        except KeyError:
            oov_count+=1
    return lwords

def disc_to_array(d, vocab, ulist):
    sub = []
    com = []
    com_ts = []
    com_user = []
    max_comment = 0
     
    sub.append(text_to_sequence(d['selftext'], vocab))
    com_ts.append(d['created_utc'])
    com_user.append(d['author'])
    for c in d['comments']:
        if list(c.values())[0]['author'] in ulist.keys():
            com.append(text_to_sequence(list(c.values())[0]['body'], vocab))
            com_ts.append(float(list(c.keys())[0]))
            com_user.append(list(c.values())[0]['author'])
            max_comment += 1
            if max_comment>500:
                break
    sub = list(pad_sequences(sub, maxlen=max_token, padding='post', truncating='post'))
    com = list(pad_sequences(com, maxlen=max_token, padding='post', truncating='post'))
    return sub, com, com_ts, com_user

if __name__=='__main__':
    voc = load_vocab('../pretrained/vocab.json')
    submissions = []
    comments = []
    comment_ts = []
    comment_users = []
    ue = load_user_embedding('../Reddit_dumps/politics/user_embedding.json')
    pred_time = load_pred_time('../Reddit_dumps/politics/pred_steps.json')
    id_file = open('../Reddit_dumps/politics/discussion_id.txt','w')
    for key, value in load_disc('../Reddit_dumps/politics/dicussions.json').items():
        if key in pred_time.keys() and value['author'] in ue.keys() and value['created_utc']>=(1512259200+10800*4):
            id_file.write(key+'\t')
            s, c, ts, cu = disc_to_array(value, voc, ue)
            submissions+=s
            comments.append(c)
            comment_ts.append(ts)
            comment_users.append(cu)
    submissions = np.array(submissions)
    id_file.close()
    print('Shape of submission array:', submissions.shape)
    np.save('../Reddit_dumps/politics/submission_input.npy', submissions)
    pickle.dump(comments, open('../Reddit_dumps/politics/comments_input.pkl','wb'), protocol=2)
    pickle.dump(comment_ts, open('../Reddit_dumps/politics/comments_ts.pkl','wb'), protocol=2)
    pickle.dump(comment_users, open('../Reddit_dumps/politics/comments_users.pkl','wb'), protocol=2)

        
    
