import json
import numpy as np
from datetime import datetime, timedelta
import os
import sys

main_dir = os.path.join("data", "reddit_data")
with open(os.path.join(main_dir, "selected_discussion_nov.jsonlist"),
          "r") as f:
    data = f.readlines()
n = len(data)
print(n)
sys.stdout.flush()

subreddit_stats = {}
input_dir_path = os.path.join("data", "reddit_data", "NOV_INPUT")
output_dir_path = os.path.join("data", "reddit_data", "NOV_OUTPUT")
output_hour_dir_path = os.path.join("data", "reddit_data", "NOV_OUTPUT_HOUR")

for i in range(n):
    print(i)
    sys.stdout.flush()
    each_reddit = json.loads(data[i])
    key = list(each_reddit.keys())[0]
    each_reddit = each_reddit[key]
    subreddit_stats[i] = {}
    subreddit_stats[i]['total'] = len(each_reddit)
    atleast_1 = 0
    atleast_10 = 0
    sub_dir_path = os.path.join(input_dir_path, str(i))
    os.mkdir(sub_dir_path)
    out_sub_dir_path = os.path.join(output_dir_path, str(i))
    os.mkdir(out_sub_dir_path)
    out_hour_sub_dir_path = os.path.join(output_hour_dir_path, str(i))
    os.mkdir(out_hour_sub_dir_path)
    for index, each_post in enumerate(each_reddit):
        event_list_temp = []
        if len(each_post['comments']) < 10:
            continue
        d1 = datetime.fromtimestamp(each_post['created_utc'])
        event_list_temp = []
        for each_comment in each_post['comments']:
            d2 = datetime.fromtimestamp(each_comment['created_utc'])
            if d2 > d1 + timedelta(days=30):
                break
            td = d2 - d1
            td = td.total_seconds() / 3600  # in hours
            event_list_temp.append(str(td) + " " + "1")
        l = len(event_list_temp)
        if l < 10:
            atleast_1 += 1
            continue
        event_list = []
        event_list.append(str(l + 1) + " " + str(each_post['created_utc']))
        event_list.append("0.0 1")
        event_list.extend(event_list_temp)
        atleast_1 += 1
        atleast_10 += 1

        file_path = os.path.join(sub_dir_path, str(index) + ".txt")
        with open(file_path, "w") as f:
            for each_line in event_list:
                f.write(each_line + "\n")
    if atleast_10 == 0:
        print("10-0", i)
        sys.stdout.flush()
        os.rmdir(sub_dir_path)
        os.rmdir(out_sub_dir_path)
        os.rmdir(out_hour_sub_dir_path)
    subreddit_stats[i]['atleast_1'] = atleast_1
    subreddit_stats[i]['atleast_10'] = atleast_10
with open(os.path.join(main_dir, "subreddit_stats_nov_10.json"), "w") as f:
    json.dump(subreddit_stats, f, indent=True)
