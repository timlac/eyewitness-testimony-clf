import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd

# from config import ROOT_DIR

#
# input_data = os.path.join(ROOT_DIR, "files/data/opensmile_functionals.csv")
# output_data = os.path.join(ROOT_DIR, "files/out")
#
# df = pd.read_csv(input_data, delimiter=";")
#
# y = df["Accuracy"].values


def plot_hist(y, title):
    plt.hist(y, bins=[0, 0.5, 1.0], edgecolor='black')
    plt.xticks([0, 1])
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Distrbution ' + title)
    plt.show()



