#!/usr/bin/env python3

# Copyright 2023 Australian Centre For Robotics
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Author: Darren Tsai

import os
import csv

import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

# NUM_BINS = 20
initial_fthresh = 2
param_fthresh = 2
path = ""


def fit_gauss(df, key, metric, num_bins):

    samples = df[key][(np.abs(stats.zscore(df[key])) < param_fthresh)]

    print("{:^5s} | Num samples after filtering: {}".format(key, len(samples)))

    mu = np.mean(samples)
    sigma = np.std(samples, ddof=1)
    pad = np.tanh(3 * sigma) * 0.2  # Use tanh to limit the padding at 0.2
    x = np.array(np.linspace(min(samples) - pad, max(samples) + pad, 100)).reshape(
        -1, 1
    )

    # Normalise and scale PDF to the same height as histogram
    h, _ = np.histogram(samples, bins=num_bins)
    y = stats.norm.pdf(x, mu, sigma) / max(stats.norm.pdf(x, mu, sigma)) * max(h)

    return {
        "key": key,
        "metric": metric,
        "samples": samples,
        "x": x,
        "y": y,
        "mu": mu,
        "sigma": sigma,
    }


def visualise_results(gauss, nbins_list, degree):

    colors = [
        "violet",
        "thistle",
        "royalblue",
        "indianred",
        "turquoise",
        "lightslategray",
    ]

    # Setup Subplot
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    fig.tight_layout(pad=3.0)  # space out the plots a bit
    fig.subplots_adjust(top=0.93)  # Adjust spacing at the top for the suptitle
    fig.suptitle("Extrinsic Parameter Results")
    row = [0, 0, 0, 1, 1, 1]
    col = [0, 1, 2, 0, 1, 2]

    for idx, param in enumerate(gauss):
        x = param["x"]
        y = param["y"]
        mu = param["mu"]
        stdev = param["sigma"]

        # Plot PDF
        r = row[idx]
        c = col[idx]
        ax[r, c].hist(
            param["samples"], bins=nbins_list[idx], alpha=0.6, color=colors[idx]
        )
        ax[r, c].plot(x, y, "-", color="black")
        ax[r, c].set_xlabel(param["key"] + " (" + param["metric"] + ")")
        ax[r, c].set_ylabel("Frequency")
        ax[r, c].set_ylim(0, max(y) + float(max(y)) / 5)
        ax[r, c].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        if degree and idx > 2:
            ax[r, c].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        else:
            ax[r, c].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        # Annotate max value of highest weighted gaussian
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(
            xycoords="data",
            textcoords="axes fraction",
            arrowprops=arrowprops,
            bbox=bbox_props,
            ha="right",
            va="top",
        )

        if degree and idx > 2:
            ax[r, c].annotate(
                param["key"] + "=% .3f\nstd=% .3f" % (mu, stdev),
                xy=(mu, max(y)),
                xytext=(0.94, 0.96),
                **kw
            )
        else:
            ax[r, c].annotate(
                param["key"] + "=% .5f\nstd=% .5f" % (mu, stdev),
                xy=(mu, max(y)),
                xytext=(0.94, 0.96),
                **kw
            )

    plt.show(block=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise calibration results")
    parser.add_argument("--csv", required=True, help="Path to calibration CSV file")
    parser.add_argument("--degree", action="store_true", help="Display rotation in degrees")
    parser.add_argument("--trans_binwidth", type=float, default=0.05, help="Translation bin width (m)")
    parser.add_argument("--rot_binwidth_deg", type=float, default=1.0, help="Rotation bin width (deg)")

    args = parser.parse_args()

    path = args.csv
    degree = args.degree
    bin_width_trans = args.trans_binwidth
    bin_width_rot = args.rot_binwidth_deg * np.pi / 180

    if not os.path.exists(path):
        raise Exception("GAUSS FITTING - No file found at: {}".format(path))

    print("Opening file at:", path)
    print("Using degrees for rotation:", degree)

    # Read data
    df_orig = pandas.read_csv(path)
    df = df_orig.copy()

    ###
    # Pre Process the file data to

    yaw_array = []
    percent_array = []
    pos_counter = 0
    neg_counter = 0

    for row in df.index:
        yaw = df["yaw"][row]

        if yaw < 0:
            neg_counter += 1
        else:
            pos_counter += 1

        # Find delta from pi
        diff = np.abs(np.pi - np.abs(yaw))
        percent = diff / np.pi * 100

        percent_array.append(percent)
        yaw_array.append(diff)

    avg = np.average(yaw_array)
    percent_avg = np.average(percent_array)

    threshold_percentage = 1.0

    if percent_avg < threshold_percentage:
        for row in df.index:
            yaw = df["yaw"][row]
            if yaw < 0:
                df.at[row, "yaw"] = 2 * np.pi + yaw
                # wrap angle between [0, 2pi]. + yaw bcs yaw is negative.

    # df.to_csv(path, index=False)

    # end of pre processing csv data

    # Initial filtering of general outliers
    params = ["roll", "pitch", "yaw", "x", "y", "z"]
    for p in params:
        df = df[(np.abs(stats.zscore(df[p])) < initial_fthresh)]

    num_bins_x = int(np.ceil((df["x"].max() - df["x"].min()) / bin_width_trans))
    num_bins_y = int(np.ceil((df["y"].max() - df["y"].min()) / bin_width_trans))
    num_bins_z = int(np.ceil((df["z"].max() - df["z"].min()) / bin_width_trans))
    num_bins_roll = int(np.ceil((df["roll"].max() - df["roll"].min()) / bin_width_rot))
    num_bins_pitch = int(
        np.ceil((df["pitch"].max() - df["pitch"].min()) / bin_width_rot)
    )
    num_bins_yaw = int(np.ceil((df["yaw"].max() - df["yaw"].min()) / bin_width_rot))
    nbins_list = [
        num_bins_roll,
        num_bins_pitch,
        num_bins_yaw,
        num_bins_x,
        num_bins_y,
        num_bins_z,
    ]
    # print(nbins_list)

    print("\nTotal number of samples: {}".format(len(df)))

    gauss = []
    if not degree:
        gauss.append(fit_gauss(df, "roll", "rad", num_bins=num_bins_roll))
        gauss.append(fit_gauss(df, "pitch", "rad", num_bins=num_bins_pitch))
        gauss.append(fit_gauss(df, "yaw", "rad", num_bins=num_bins_yaw))
    else:
        df["roll"] = df_orig["roll"] * 180 / np.pi
        df["pitch"] = df_orig["pitch"] * 180 / np.pi
        df["yaw"] = df_orig["yaw"] * 180 / np.pi
        gauss.append(fit_gauss(df, "roll", "deg", num_bins=num_bins_roll))
        gauss.append(fit_gauss(df, "pitch", "deg", num_bins=num_bins_pitch))
        gauss.append(fit_gauss(df, "yaw", "deg", num_bins=num_bins_yaw))

    gauss.append(fit_gauss(df, "x", "m", num_bins=num_bins_x))
    gauss.append(fit_gauss(df, "y", "m", num_bins=num_bins_y))
    gauss.append(fit_gauss(df, "z", "m", num_bins=num_bins_z))

    print("\n")
    visualise_results(gauss, nbins_list, degree)

plt.show()