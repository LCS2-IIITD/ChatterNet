import json
from_subred_file = open('../Reddit dumps/Subreddit_list.txt').read().split('\n')[:-1]
selected_subred = []
for s in from_subred_file:
    if s[-1]=='*':
        selected_subred.append(s.split(':')[0])
#selected_subred = ['Futurology','technology', 'The_Donald','politics', 'science', 'news', 'worldnews']

comment_rate = []
start = 1541030400
subreddit_comment_ts = {}
for subred in selected_subred:
    subreddit_comment_ts[subred]=[]
with open('../Reddit dumps/Comments/RC_2018-11') as infile:
    for line in infile:
        comment = json.loads(line)
        subred = comment['subreddit']
        ts = comment['created_utc']
        if subred in subreddit_comment_ts.keys():
            subreddit_comment_ts[subred].append(ts)
subred_comment_rate = {}
for subred, comments in subreddit_comment_ts.items():
    end = start + 60
    comment_array = []
    index = 0
    comments = sorted(comments)
    for com in comments:
        if not index:
            count = 0
        if com>=start:
            if com<end and index<len(comments)-1:
                count+=1
            else:
                comment_array.append(count)
                end += 60
                count = 1
        index+=1
    subred_comment_rate[subred] = comment_array

with open('comment_rate_november.json','w') as outfile:
    json.dump(subred_comment_rate, outfile)
