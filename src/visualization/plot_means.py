import matplotlib.pyplot as plt

import os

import pandas as pd
import numpy as np

from src.preprocessing.normalization.functional_normalization import functional_normalize_by
from config import ROOT_DIR, AUDIO_FUNCTIONALS_EGEMAPS_COLS, AUDIO_VISUALIZATION_COLS


def plot_means_and_stds(x, y, cols, subtitle):
    for idx, col in enumerate(cols):
        means = {}
        stds = {}

        for accuracy in (0, 1):
            indices = np.where(y == accuracy)[0]
            x_set = x[indices]
            x_col = x_set[:, idx]
            means[str(accuracy)] = np.mean(x_col)
            stds[str(accuracy)] = np.std(x_col)

        # plt.figure(figsize=(15, 10))
        plt.errorbar(means.keys(), means.values(), yerr=list(stds.values()),
                     fmt='o', ecolor="black", capsize=2, elinewidth=1)
        plt.title(col + " " + subtitle)
        # plt.xticks(rotation=90, fontsize=15)
        # plt.yticks(fontsize=15)
        plt.ylabel("mean and std")
        plt.show()


def main():
    df = pd.read_csv(os.path.join(ROOT_DIR, "data/opensmile_functionals.csv"), delimiter=";")

    y = df["Accuracy"].values
    x = df[AUDIO_VISUALIZATION_COLS].values
    participant = df["Participant"].values

    plot_means_and_stds(x, y, AUDIO_VISUALIZATION_COLS, "pre")

    x_normalized = functional_normalize_by(x=x, identifiers=participant, method="min_max")

    plot_means_and_stds(x_normalized, y, AUDIO_VISUALIZATION_COLS, "post")


main()