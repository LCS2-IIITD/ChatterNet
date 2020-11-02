import json
id_to_ts = {}
with open('selected_discussion_nov.jsonlist') as infile:
    for line in infile:
        d = json.loads(line)
        for disc in list(d.values())[0]:
            ts_list = []
            ts_list.append(0)
            for com in disc['comments']:
                ts_list.append(com['created_utc']-disc['created_utc'])
            id_to_ts[disc['id']] = ts_list
            
with open('disc_id2ts_nov.json','w') as outfile:
    json.dump(id_to_ts, outfile)
