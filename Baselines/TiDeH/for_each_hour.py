import argparse
import os
import json
import sys
from tideh import functions
import numpy as np
import sys


def load_events(filename, time_factor=1, start_factor=1):
    """
    Loads events form given file path.
    :param filename: path to file
    :param time_factor: factor to multiply time with, useful to convert time unit
    :param start_factor: factor to multiply start_time with
    :return: tuple, first element contains tuple of number of events and start time of observation, second elements
    holds event as array of tuple (event_time, number_of_followers)
    """
    res = []

    with open(filename, "r") as in_file:
        first = next(in_file)
        values_first = first.split(" ")
        for line in in_file:
            values = line.split(" ")
            res.append((float(values[0]) * time_factor, int(values[1])))

    return (float(values_first[0]), float(values_first[1]) * start_factor), res


def load_events_vec(filename, time_factor=1, start_factor=1):
    """
    Loads events in shape of a tuple of nd-arrays.

    The returned values can be used as input for all optimized functions.

    :param filename: path to file
    :param time_factor: factor to multiply time with, useful to convert time unit
    :param start_factor: factor to multiply start_time with
    :return: 2-tuple, first element holding tuple of number of events and start time, second holding tuple of event
    times and follower nd-arrays
    """
    (nb_events, start_time), events = load_events(filename, time_factor,
                                                  start_factor)
    event_times = np.array([e[0] for e in events])
    followers = np.array([e[1] for e in events])
    return (nb_events, start_time), (event_times, followers)


def estimate_infectious_rate_vec(event_times,
                                 follower,
                                 kernel_integral=functions.integral_zhao_vec,
                                 obs_time=1,
                                 pred_time=720,
                                 window_size=1,
                                 window_stride=1):
    """
    Estimates infectious rate using moving time window approach.
    Optimized using numpy and vectorized approach.

    :param event_times: nd-array of event times
    :param follower: nd-array of follower counts
    :param kernel_integral: function for calculating the integral of the kernel
    :param obs_time: observation time
    :param window_size: bin width for estimation (in hours)
    :param window_stride: interval for moving windows (in hours)
    :return: 3-tuple holding list of estimated infectious rate for every moving time window, event counts for every
    window and the time in the middle of every window
    """
    estimations = []
    window_middle = []
    window_event_count = []

    for start in range(obs_time, pred_time - window_size + window_stride,
                       window_stride):
        end = start + window_size
        mask = event_times < end  # all events up until end of current interval
        count_current = get_event_count(event_times, start, end)
        est = estimate_infectious_rate_constant_vec(
            event_times[mask],
            follower[mask],
            t_start=start,
            t_end=end,
            kernel_integral=kernel_integral,
            count_events=count_current)

        window_middle.append(start + window_size / 2)
        window_event_count.append(count_current)
        estimations.append(est)
    return estimations, window_event_count, window_middle


def get_event_count(event_times, start, end):
    """
    Count of events in given interval.

    :param event_times: nd-array of event times
    :param start: interval start
    :param end: interval end
    :return: count of events in interval
    """
    mask = (event_times > start) & (event_times <= end)
    return event_times[mask].size


def estimate_infectious_rate_constant_vec(event_times,
                                          follower,
                                          t_start,
                                          t_end,
                                          kernel_integral,
                                          count_events=None):
    """
    Returns estimation of infectious rate for given event time and followers on defined interval.
    Optimized using numpy.

    :param event_times: nd-array of event times
    :param follower: nd-array of follower counts
    :param t_start: time interval start
    :param t_end: time interval end
    :param kernel_integral: integral function of kernel function
    :param count_events: count of observed events in interval (used for time window approach)
    :return: estimated values for infectious rate
    """
    kernel_integral_ = kernel_integral(t_start - event_times,
                                       t_end - event_times)
    kernel_int = follower * kernel_integral_

    if count_events is not None:
        return count_events / kernel_int.sum()
    else:
        return event_times.size / kernel_int.sum()


parser = argparse.ArgumentParser(description='Input Params')
parser.add_argument('--m', help='month, either oct or nov')
parser.add_argument('--srd', help='subreddit_number')
parser.add_argument('--fl', help='file_number')
parser.add_argument('--ot', help='observation time in hrs')
parser.add_argument('--pt', help='prediction time in hrs')

args = parser.parse_args()
month = args.m if args.m else 'nov'
month = month.upper()
input_path = os.path.join("data", "reddit_data", month + "_INPUT")
output_path = os.path.join("data", "reddit_data", month + "_OUTPUT_HOUR")

subreddit_number = args.srd
filename = args.fl + ".txt"
filename = os.path.join(input_path, subreddit_number, filename)
obs_time = int(args.ot) if args.ot else 1
pred_time = int(args.pt) if args.pt else 720  # 24 * 30

(_, start_time), (event_times, follower) = load_events_vec(filename)

window_size = 1  # one hour each (hourly rate).
window_stride = 1  # one hour each (hourly rate).
filt_obs = obs_time <= event_times
filt_obs = event_times <= pred_time
estimations, window_event_count, _ = estimate_infectious_rate_vec(
    event_times=event_times[filt_obs],
    follower=follower[filt_obs],
    kernel_integral=functions.integral_zhao_vec,
    obs_time=obs_time,
    pred_time=pred_time,
    window_size=window_size,
    window_stride=window_stride)

results = {}
results["estimations"] = estimations
results["window_event_count"] = window_event_count

filename = args.fl + ".json"
output_file = os.path.join(output_path, subreddit_number, filename)
with open(output_file, "w") as f:
    json.dump(results, f, indent=True)