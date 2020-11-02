import os
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse

parser = argparse.ArgumentParser(description='Input Params')
parser.add_argument('--m', help='month, either oct or nov')
args = parser.parse_args()
month = args.m if args.m else 'nov'

main_dir = os.path.join("data", "reddit_data")
input_dir = os.path.join(main_dir, month.upper() + "_INPUT")
output_dir = os.path.join(main_dir, month.upper() + "_OUTPUT")

folders = []
bin_data = []
for r, _, _ in os.walk(input_dir):
    folders.append(r)
folders = folders[1:]
for each_sub in folders:
    sub_red = each_sub.split("/")[-1]
    for _, _, f in os.walk(os.path.join(input_dir, sub_red)):
        file_list = f
    for each_file in file_list:
        data_tuple = []
        with open(os.path.join(input_dir, sub_red, each_file), "r") as f:
            data = f.readline()
        data_tuple.append(int(data.split(" ")[1].strip()))
        file_number = each_file.split(".txt")[0]
        with open(os.path.join(output_dir, sub_red, file_number + ".json"),
                  "r") as f:
            data = json.load(f)
        data_tuple.append(data["from_to_pred"])
        data_tuple.append(data["original_to_pred"])
        bin_data.append(data_tuple)

bins = {i: [] for i in range(1, 31)}
for i in bin_data:
    date_ = datetime.fromtimestamp(i[0])
    b = int(date_.day)
    bins[b].append(i)

mse_bin = {i: 0 for i in range(1, 31)}
for b in bins:
    data_list = bins[b]
    y_pred = []
    y_ground = []
    for data in data_list:
        y_pred.append(data[1] + 1.0)
        y_ground.append(data[2] + 1.0)
    n_pred = np.array(y_pred)
    n_ground = np.array(y_ground)
    mse = mean_squared_error(np.log(n_ground), np.log(n_pred))
    mse_bin[b] = mse

with open(os.path.join(main_dir, "evaluation_mse_bins_" + month + ".json"), "w") as f:
    json.dump(mse_bin, f, indent=True)