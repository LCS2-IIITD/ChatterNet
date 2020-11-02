import os, sys
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import argparse

parser = argparse.ArgumentParser(description='Input Params')
parser.add_argument('--m', help='month, either oct or nov')
args = parser.parse_args()
month = args.m if args.m else 'nov'

main_dir = os.path.join("data", "reddit_data")
output_dir = os.path.join(main_dir, month.upper() + "_OUTPUT_HOUR")
folders = []
l_es = []
l_wc = []
for r, _, _ in os.walk(output_dir):
    folders.append(r)
folders = folders[1:]
for each_sub in folders:
    print(each_sub)
    sub_red = each_sub.split("/")[-1]
    for _, _, f in os.walk(os.path.join(output_dir, sub_red)):
        file_list = f
    for each_file in file_list:
        with open(os.path.join(output_dir, sub_red, each_file), "r") as f:
            data = json.load(f)
        es = data['estimations']
        wc = data['window_event_count']
        for e, w in zip(es, wc):
            if e == 0 and w == 0:
                break
            l_es.append(int(e) + 1.0)
            l_wc.append(int(w) + 1.0)

np_es = np.array(l_es)
np_wc = np.array(l_wc)
mse = mean_squared_error(np.log(np_wc), np.log(np_es))
tau, _ = stats.kendalltau(np_wc, np_es)
spr, _ = stats.spearmanr(np_wc, np_es)
results = {}
results["mse_rate"] = mse
results["tau_rate"] = tau
results["spr_rate"] = spr
result_filename = os.path.join(main_dir, "evaluation_rates_" + month + ".json")
with open(result_filename, "w") as f:
    json.dump(results, f, indent=True)
print(mse, tau, spr)
