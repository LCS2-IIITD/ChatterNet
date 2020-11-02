"""
Example of predicting future retweet activity using functions implemented in native Python.
First the parameters of the infectious rate are estimated for the given observation time. Then, the number of retweets
before a given time (= pred_time) is predicted.

Inputs are
  1) Data file that includes the retweet times and the number of followers
     Here, this code reads 'data/example/sample_file.txt' (= filename).
  2) Observation time (= obs_time).
  3) Final time of prediction (= pred_time).

Outputs are
  1) Estimate of model parameters of TiDeH (p_0, r_0, phi_0, t_m).
  2) Number of retweets from obs_time (h) to pred_time (h).

You can change the window size for estimation (\delta_obs) by calling the function "estimate_parameters" as following:
params, err, _ = estimate_parameters(events=events, obs_time=obs_time, window_size=window_size, **add_params)
where the variable "window_size" represents \delta_obs.

You can also change the window size for prediction (\delta_pred) by calling the function "predict" as following:
_, total, pred_error = predict(events=events, obs_time=obs_time, pred_time=pred_time, window=window, p_max=None,
                               params=params, **add_params)
where the variable "window" represents \delta_pred.

This code is developed by Sylvain Gauthier and Sebastian Rühl under the supervision of Ryota Kobayashi.
"""
from tideh import estimate_parameters
from tideh import load_events
from tideh import predict
import argparse
import os
import json
import sys

input_path = os.path.join("data", "reddit_data", "OCT_INPUT")
output_path = os.path.join("data", "reddit_data", "OCT_OUTPUT")

parser = argparse.ArgumentParser(description='Input Params')
parser.add_argument("--srd", help='subreddit_number')
parser.add_argument('--fl', help='file_number')
parser.add_argument('--ot', help='observation time in hrs')
parser.add_argument('--pt', help='prediction time in hrs')

args = parser.parse_args()
subreddit_number = args.srd
filename = args.fl + ".txt"
filename = os.path.join(input_path, subreddit_number, filename)
obs_time = int(args.ot) if args.ot else 24
pred_time = int(args.pt) if args.pt else 744  # 24 * 31

# filename = 'data/example/sample_file.txt'
# obs_time = 24  # observation time of 1 day
# pred_time = 24 * 31  # predict for one month

# the number of retweets is not necessary for the further steps
# make sure that all times are loaded in the correct time unit (hours)
(_, start_time), events = load_events(filename)
# additional parameters passed to infectious rate function
add_params = {'t0': start_time, 'bounds': [(-1, 0.5), (1, 20.)]}

params, err, _ = estimate_parameters(
    events=events,
    obs_time=obs_time,
    # pred_time=pred_time,
    **add_params)
results = {}
results['p0'] = params[0]
results['r0'] = params[1]
results['phi0'] = params[2]
results['tm'] = params[3]
results['avg_fit_error'] = err * 100

print("Estimated parameters are:")
sys.stdout.flush()
print("p0:   %.3f" % params[0])
sys.stdout.flush()
print("r0:   %.3f" % params[1])
sys.stdout.flush()
print("phi0: %.3f" % params[2])
sys.stdout.flush()
print("tm:   %.3f" % params[3])
sys.stdout.flush()
print("Average %% error (estimated to fitted): %.2f" % (err * 100))
sys.stdout.flush()
# predict future retweets
_, total, pred_error, pred_original_length = predict(events=events,
                                                     obs_time=obs_time,
                                                     pred_time=pred_time,
                                                     p_max=None,
                                                     params=params,
                                                     **add_params)

results['from_to_pred'] = int(total)
results['original_to_pred'] = pred_original_length
total_pred = len([e for e, _ in events if e <= obs_time]) + total
results['total_pred'] = total_pred
results["pred_abs_error"] = pred_error

print("Predicted number of retweets from %s to %s hours: %i" %
      (obs_time, pred_time, total))
sys.stdout.flush()
print("Predicted number of retweets at hour %s: %i" % (pred_time, total_pred))
sys.stdout.flush()
print("Prediction error (absolute): %.0f" % pred_error)
sys.stdout.flush()

filename = args.fl + ".json"
output_file = os.path.join(output_path, subreddit_number, filename)
with open(output_file, "w") as f:
    json.dump(results, f, indent=True)
print("End")