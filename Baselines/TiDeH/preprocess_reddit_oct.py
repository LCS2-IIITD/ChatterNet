import json
import numpy as np
from datetime import datetime, timedelta
import os

main_dir = os.path.join("data", "reddit_data")
with open(os.path.join(main_dir, "selected_discussion_oct.jsonlist"), "r") as f:
    data = f.readlines()
n = len(data)
print(n)

subreddit_stats = {}
input_dir_path = os.path.join("data", "reddit_data", "OCT_INPUT")
output_dir_path = os.path.join("data", "reddit_data", "OCT_OUTPUT")
output_hour_dir_path = os.path.join("data", "reddit_data", "NOV_OUTPUT_HOUR")

for i in range(n):
    print(i)
    each_reddit = json.loads(data[i])
    each_reddit = each_reddit['subreddit']
    subreddit_stats[i] = {}
    subreddit_stats[i]['total'] = len(each_reddit)
    subreddit_stats[i]['atleast_1'] = sum(
        [1 for each_post in each_reddit if len(each_post['comments']) > 0])
    subreddit_stats[i]['atleast_10'] = sum(
        [1 for each_post in each_reddit if len(each_post['comments']) >= 10])
    if subreddit_stats[i]['atleast_10'] == 0:
        continue
    sub_dir_path = os.path.join(input_dir_path, str(i))
    os.mkdir(sub_dir_path)
    out_sub_dir_path = os.path.join(output_dir_path, str(i))
    os.mkdir(out_sub_dir_path)
    out_hour_sub_dir_path = os.path.join(output_hour_dir_path, str(i))
    os.mkdir(out_hour_sub_dir_path)
    for index, each_post in enumerate(each_reddit):
        event_list = []
        if len(each_post['comments']) < 10:
            continue
        event_list.append(
            str(len(each_post['comments']) + 1) + " " +
            str(each_post['created_utc']))
        d1 = datetime.fromtimestamp(each_post['created_utc'])
        event_list.append("0.0 1")
        for each_comment in each_post['comments']:
            d2 = datetime.fromtimestamp(each_comment['created_utc'])
            td = d2 - d1
            td = td.total_seconds() / 3600  # in hours
            event_list.append(str(td) + " " + "1")
        file_path = os.path.join(sub_dir_path, str(index) + ".txt")
        with open(file_path, "w") as f:
            for each_line in event_list:
                f.write(each_line + "\n")

with open(os.path.join(main_dir, "subreddit_stats_10.json"), "w") as f:
    json.dump(subreddit_stats, f, indent=True)
