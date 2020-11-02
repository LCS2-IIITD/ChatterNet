import json
import os
import datetime
from datetime import datetime, timedelta
import random
import sys
import networkx as nx


def return_mapping(raw_id):
    id_key = poster_ids.get(raw_id, -1)
    if id_key == -1:
        id_key = commenter_ids[raw_id]
    if id_key == "[deleted]":
        id_key = raw_id
    return user_count_mapping[id_key]


def updated_mapping(raw_id):
    global id_count
    if raw_id not in user_count_mapping:
        user_count_mapping[raw_id] = id_count
        count_user_mapping[id_count] = raw_id
        id_count += 1
    return user_count_mapping[raw_id]

# Jsonlist of Reddit Data
file_path = "../TiDeH/data/reddit_data/selected_discussion_nov.jsonlist"
with open(file_path, "r") as f:
    data = f.readlines()
n = len(data)
print(n)
sys.stdout.flush()

# Get id to username mapping of a post and a comment.
with open("../TiDeH/data/reddit_data/post_author.json", "r") as f:
    poster_ids = json.load(f)
with open("../TiDeH/data/reddit_data/comment_author.json", "r") as f:
    commenter_ids = json.load(f)

print(len(poster_ids))
print(len(commenter_ids))
sys.stdout.flush()

user_count_mapping = {}
count_user_mapping = {}
id_count = 0
g = nx.Graph()

for i in range(n):
    print(i)
    sys.stdout.flush()
    each_reddit = json.loads(data[i])
    key = list(each_reddit.keys())[0]
    each_reddit = each_reddit[key]
    for each_post in each_reddit:
        if len(each_post['comments']) < 10: # Only cascades of length 10 or more are considered for the experiment.
            continue
        user_id = each_post['id']
        real_name = poster_ids[user_id]
        if real_name == '[deleted]': #All deleted ids are mapped to a single unique id.
            user_id_count = updated_mapping(user_id)
        else:
            user_id_count = updated_mapping(real_name)
        #prev_comments = []
        for each_comment in each_post['comments']:
            comment_id = each_comment['id']
            real_name = commenter_ids[comment_id]
            if real_name == '[deleted]':
                comment_id_count = updated_mapping(comment_id)
            else:
                comment_id_count = updated_mapping(real_name)

            if g.has_edge(user_id_count, comment_id_count): # add edge between posters and commenters.
                g[user_id_count][comment_id_count]['weight'] += 1
            else:
                g.add_edge(user_id_count, comment_id_count, weight=1)

            #for each_edge in prev_comments:
             #   if g.has_edge(each_edge, comment_id_count):
              #      g[each_edge][comment_id_count]['weight'] += 0.25
               # else:
                #    g.add_edge(each_edge, comment_id_count, weight=0.25)
            #prev_comments.append(comment_id_count)

assert len(user_count_mapping) == g.number_of_nodes()

graph_line = []
for each_node in g.nodes():
    line = ""
    line = str(each_node) + "\t\t"
    flag = False
    for each_neighbor in g.neighbors(each_node):
        flag = True
        weight = g[each_node][each_neighbor].get('weight', 0.0)
        line += str(each_node) + ":" + str(each_neighbor) + ":" + str(
            weight) + "\t"
    if flag == False:
        line += None + "\t"
    line += "\n"
    graph_line.append(line)

with open("global_graph.txt", "w") as f:
    for each_line in graph_line:
        f.write(each_line)

cascade_line = []
cas_id = 0
print("________CASCADE____")
for i in range(n):
    print(i)
    sys.stdout.flush()
    each_reddit = json.loads(data[i])
    key = list(each_reddit.keys())[0]
    each_reddit = each_reddit[key]
    for each_post in each_reddit:
        count_delta_1 = 0 # delta_t
        count_delta_10 = 0 # delta_t
        count_delta_30 = 0 # delta_t 
        count_1_hr = 0 # t
        d1 = datetime.fromtimestamp(each_post['created_utc'])
        if len(each_post['comments']) < 10:
            continue
        cas_line = ""
        user_id_count = return_mapping(each_post['id'])
        cas_line = str(cas_id) + "\t" + str(user_id_count) + "\t2009\t"
        comment_str = ""
        for each_comment in each_post['comments']:
            comment_id_count = return_mapping(each_comment['id'])
            d2 = datetime.fromtimestamp(each_comment['created_utc'])
            if d2 > d1 + timedelta(days=30):
                break
            if d2 < d1 + timedelta(hours=1):
                count_1_hr += 1
                weight = g[user_id_count][comment_id_count].get('weight', 0.0)
                comment_str += str(user_id_count) + ":" + str(
                    comment_id_count) + ":" + str(weight) + " "
            else:
                if d2 <= d1 + timedelta(days=1):
                    count_delta_1 += 1
                    count_delta_10 += 1
                    count_delta_30 += 1
                elif d2 <= d1 + timedelta(days=10):
                    count_delta_10 += 1
                    count_delta_30 += 1
                elif d2 <= d1 + timedelta(days=30):
                    count_delta_30 += 1
        if count_1_hr == 0:
            continue
        else:
            cas_line += str(count_1_hr) + "\t" + comment_str + "\t" + str(
                count_delta_1) + " " + str(count_delta_10) + " " + str(
                    count_delta_30) + "\n"
            cascade_line.append(cas_line)
            cas_id += 1

print(len(cascade_line))
sys.stdout.flush()
p = int(len(cascade_line) * 0.05)

random.shuffle(cascade_line)
random.shuffle(cascade_line)
random.shuffle(cascade_line)

with open("cascade_val.txt", "w") as f:
    for each_line in cascade_line[:p]:
        f.write(each_line)

with open("cascade_test.txt", "w") as f:
    for each_line in cascade_line[p:3 * p]:
        f.write(each_line)

with open("cascade_train.txt", "w") as f:
    for each_line in cascade_line[3 * p:]:
        f.write(each_line)
print("----END___")
sys.stdout.flush()
