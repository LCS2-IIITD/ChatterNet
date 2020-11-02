import json
from multiprocessing import Process
outputjson = open('selected_discussion_oct.jsonlist','w')

def select_subreddits():
    from_subred_file = open('../Reddit dumps/Subreddit_list.txt').read().split('\n')[:-1]
    selected_subred = []
    subredditfile = open('sublist.txt','w')
    for s in from_subred_file:
        if s[-1]=='*':
            selected_subred.append(s.split(':')[0])
            subredditfile.write(s.split(':')[0]+'\t\t')
    subredditfile.close()
    return selected_subred

def select_submissions(subreddits):
    subreddit_submission_dict = {}
    for subreddit in subreddits:
        subreddit_submission_dict[subreddit] = []
    with open('../Reddit dumps/Submissions/RS_2018-10') as infile:
        for line in infile:
            c = json.loads(line)
            if c['created_utc']>1538611200 and c['subreddit'] in subreddit_submission_dict.keys(): #
                c_ = {}
                c_['author'] = c['author']
                c_['title'] = c['title']
                c_['selftext'] = c['selftext']
                c_['created_utc'] = c['created_utc']
                c_['id'] = c['id']
                subreddit_submission_dict[c['subreddit']].append(c_)
    for s in subreddit_submission_dict.keys():
        subreddit_submission_dict[s] = sorted(subreddit_submission_dict[s], key = lambda i:i['created_utc'])
    return subreddit_submission_dict
def select_comments(subreddits):
    subreddit_comments_dict = {}
    for subreddit in subreddits:
        subreddit_comments_dict[subreddit] = []
    with open('../Reddit dumps/Comments/RC_2018-10') as infile:
        for line in infile:
            c = json.loads(line)
            if c['created_utc']>1538611200 and c['subreddit'] in subreddit_comments_dict.keys():
                c_ = {}
                c_['author'] = c['author']
                c_['link_id'] = c['link_id']
                c_['body'] = c['body']
                c_['created_utc'] = c['created_utc']
                c_['id'] = c['id']
                c_['parent_id'] = c['parent_id']
                subreddit_comments_dict[c['subreddit']].append(c_)
    with open('../Reddit dumps/Comments/RC_2018-11') as infile:
        for line in infile:
            c = json.loads(line)
            if c['created_utc']>1538611200 and c['subreddit'] in subreddit_comments_dict.keys():
                c_ = {}
                c_['author'] = c['author']
                c_['link_id'] = c['link_id']
                c_['body'] = c['body']
                c_['created_utc'] = c['created_utc']
                c_['id'] = c['id']
                c_['parent_id'] = c['parent_id']
                subreddit_comments_dict[c['subreddit']].append(c_)
    return subreddit_comments_dict

def create_discussion(submission_list, comment_list, subreddit):
    sub_id_dict = {}
    for s in submission_list:
        s['comments'] = []
        sub_id_dict[s['id']] = s

    for c in comment_list:
        link_id = c['link_id'].split('_')[-1]
        try:
            sub_id_dict[link_id]['comments'].append(c)
        except KeyError:
            pass

    threads = list(sub_id_dict.values())
    global outputjson
    outputjson.write('{}\n'.format(json.dumps({'subreddit':threads})))

if __name__=='__main__':
    subreddits = select_subreddits()
    #submissions = select_submissions(subreddits)
    #print('Finished submission selection...')
    #with open('../temp_Submissions.json','w') as tout:
    #    json.dump(submissions, tout)
    #del submissions
    #comments = select_comments(subreddits)
    #print('Finished comment selection...')
    
    #with open('../temp_Comments.json','w') as tout:
    #    json.dump(comments, tout)
    #print('Starting discussion construction...')

    #for subreddit in subreddits:
    #    create_discussion(submissions[subreddit], comments[subreddit],subreddit)
    
    #outputjson.close()

                
