import json
import os
import numpy as np
import scipy.stats as stats

delta_t = 1
print(delta_t)
with open("../data/test-net/nov_test"+str(delta_t)+".json","r") as f:
	results = json.load(f)
error={}

error["delta_t"]=delta_t
error["kendall_overall"]=0.0
error["spearman_overall"]=0.0
l_overall_pred=[]
l_overall_ground=[]
for each in zip(results["predicted_y"],results["original_y"]):
	y_pred = each[0]
	y_ground = each[1]
        if y_pred==0 and y_ground==0:
            continue
        elif y_ground==0:
            y_ground=1.0000001
        l_overall_pred.append(y_pred)
	l_overall_ground.append(y_ground)
n_pred = np.array(l_overall_pred)
n_ground = np.array(l_overall_ground)
p_err = sum(np.absolute(n_ground - n_pred)/n_ground) / len(n_ground)
tau, _ = stats.kendalltau(n_ground, n_pred)
print("t",tau)
spr, _ = stats.spearmanr(n_ground, n_pred)
print("s",spr)
print("log_abs",p_err)
error['kendall_overall'] = tau
error['spearman_overall'] = spr
error['percent_log_abs'] = p_err #Already in log scale

with open("../data/test-net/evaluation_nov"+str(delta_t)+".json","w") as f:
	    json.dump(error, f, indent=True)
