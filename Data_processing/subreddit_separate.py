import json

sub_dict = {}
with open('../Reddit dumps/Subreddit_list.txt') as infile:
    for line in infile:
        sub_dict[line.split(':')[0]] = []
with open('../Reddit dumps/Submissions/RS_2018-11') as infile:
    for line in infile:
        sub = json.loads(line)
        if sub['subreddit'] in sub_dict.keys():
            d = {}
            d['id'] = sub['id']
            d['created_utc'] = sub['created_utc']
            d['text'] = sub['title']+'\n'+sub['selftext']
            sub_dict[sub['subreddit']].append(d)

com_dict = {}
for key in sub_dict.keys():
    com_dict[key] = {}
with open('../Reddit dumps/Comments/RC_2018-11') as infile:
    for line in infile:
        com = json.loads(line)
        if com['subreddit'] in com_dict.keys():
            if com['link_id'].split('_')[-1] not in com_dict[com['subreddit']].keys():
                com_dict[com['subreddit']][com['link_id'].split('_')[-1]] = []
            com_dict[com['subreddit']][com['link_id'].split('_')[-1]].append(com['created_utc'])

with open('../Reddit dumps/Comments/RC_2018-12') as infile:
    for line in infile:
        com = json.loads(line)
        if com['subreddit'] in com_dict.keys():
            if com['link_id'].split('_')[-1] not in com_dict[com['subreddit']].keys():
                com_dict[com['subreddit']][com['link_id'].split('_')[-1]] = []
            com_dict[com['subreddit']][com['link_id'].split('_')[-1]].append(com['created_utc'])

for key, value in sub_dict.items():
    for d in value:
        if d['id'] in com_dict[key].keys():
            count = len([x for x in com_dict[key][d['id']] if (x-sub_dict[key][value.index(d)]['created_utc'])<=3600*24])
            sub_dict[key][value.index(d)]['comment_count'] = count

with open('../Reddit dumps/selected_submissions_november.json','w') as outfile:
    json.dump(sub_dict, outfile)
