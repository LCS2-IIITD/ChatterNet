"""I
Author: Sarah Masud
Copyright (c): Sarah Masud
"""

import os
import json
import numpy as np
import scipy.stats as stats
import argparse

parser = argparse.ArgumentParser(description='Input Params')
parser.add_argument('--dt', help='month, either oct or nov')
args = parser.parse_args()
delta_t = args.dt

main_dir = os.path.join("../","data","test-net")
file_name = os.path.join(main_dir,"nov_test"+delta_t+".json")
k = 10
bins = np.log((np.array([0.00001, 2,3, 4, 5, np.inf])*k))
l_pred=[]
l_ground=[]
with open(file_name, "r") as f:
    results = json.load(f)
pred_count = results['predicted_y']
ground_count = results['original_y']
m = len(bins)
for pred in pred_count:
    for j in range(m-1):
        if bins[j]<=pred<bins[j+1]:
            l_pred.append(bins[j])
            break

for ground in ground_count:
    for j in range(m-1):
        if bins[j]<=ground<bins[j+1]:
            l_ground.append(bins[j])
            break
np_ground = np.array(l_ground).astype(float)
np_pred = np.array(l_pred).astype(float)
tau, _ = stats.kendalltau(np_ground, np_pred)
spr, _ = stats.spearmanr(np_ground, np_pred)
results = {}
results["tau_cascade_overall"] = tau
results["spr_cascade_overall"] = spr
results["delta_t"] = delta_t
result_filename = os.path.join(main_dir,
                               "evaluation_cascade_k" + str(k) +"_delta_"+delta_t +".json")
with open(result_filename,"w") as f:
	json.dump(results, f, indent=True)

print("tau", tau)
print("spr", spr)
