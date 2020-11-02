from datetime import datetime
import json
##from_subred_file = open('../Reddit dumps/Subreddit_list.txt').read().split('\n')[:-1]
##selected_subred = []
##for s in from_subred_file:
##    if s[-1]=='*':
##        selected_subred.append(s.split(':')[0])
ti = []
##with open('../Reddit dumps/Submissions/RS_2018-10') as infile:
##    for line in infile:
##        sub = json.loads(line)
##        if sub['subreddit'] in selected_subred:
##            ti.append(sub['created_utc'])
with open('../News_articles/cc_news_en.jsonlist') as infile:
    for line in infile:
        sub = json.loads(line)
        date_obj = datetime.strptime(sub['date_publish'], '%Y-%m-%d %H:%M:%S')
        ts = round(date_obj.timestamp() + (datetime.now().timestamp() -datetime.utcnow().timestamp()))
        ti.append(ts)
ti = sorted(ti)
start = ti[0]+60
count = 0
frame=[]
pfile = open('temporary_news_count.txt','w')
for t in ti:
    if len(frame)>15000:
        break
    if t<start:
        count+=1
    else:
        frame.append(count)
        pfile.write(str(count)+'\t')
        count=1
        start=start+60
pfile.close()
