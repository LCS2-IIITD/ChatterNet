import json
import os
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser(description='Input Params')
parser.add_argument('--m', help='month, either oct or nov')
args = parser.parse_args()
month = args.m if args.m else 'nov'

# sub_reddit_list = ['1','2',<list of folder number>]
sub_reddit_list = os.listdir("data/reddit_data/" + month.upper() + "_INPUT")
print("###################################")
print("####################################")
print(sub_reddit_list)
print("\n\n")
sys.stdout.flush()
main_dir = os.path.join("data", "reddit_data")
input_dir = os.path.join(main_dir, month.upper() + "_INPUT")
str_name = ''
error_list = []

# ToDo: Can be parallized at subreddit or post level.
try:
    for sub_red in sub_reddit_list:
        print("######################################")
        print("######################################")
        print(sub_red)
        sys.stdout.flush()
        str_name += sub_red + "_"
        for _, _, f in os.walk(os.path.join(input_dir, sub_red)):
            file_list = f
        file_list = [f.split(".txt")[0] for f in file_list]
        for each_post in file_list:
            print(each_post)
            sys.stdout.flush()
            cmd = "python3 example_optimized.py --m {0} --srd {1} --fl {2}".format(
                month, sub_red, each_post)
            returned = subprocess.call(cmd, shell=True)
            if returned:
                error_list.append((sub_red, each_post))

except Exception:
    pass

finally:
    err_filename = os.path.join(main_dir, str_name + "error_list.json")
    with open(err_filename, "w") as f:
        json.dump(error_list, f, indent=True)
