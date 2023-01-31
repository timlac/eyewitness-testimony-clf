import os
import json

import pandas as pd
import numpy as np
from sklearn.svm import SVC


from src.preprocessing.normalization.functional_normalization import functional_normalize_by
from src.learning.evaluate_scores import evaluate_scores
from src.learning.parameter_tuning import ParamSearch
from config import ROOT_DIR, AUDIO_FUNCTIONALS_EGEMAPS_COLS, AUDIO_VISUALIZATION_COLS
from src.visualization.plot_means import plot_means_and_stds

# paths
input_data = os.path.join(ROOT_DIR, "files/data/opensmile_functionals.csv")
output_data = os.path.join(ROOT_DIR, "files/out")

df = pd.read_csv(input_data, delimiter=";")

y = df["Accuracy"].values
x = df[AUDIO_FUNCTIONALS_EGEMAPS_COLS].values
participant = df["Participant"].values

plot_means_and_stds(x, y, AUDIO_VISUALIZATION_COLS, "pre")

x = functional_normalize_by(x, participant, method="min_max")

plot_means_and_stds(x, y, AUDIO_VISUALIZATION_COLS, "post")


params_path = os.path.join(output_data, "best_params.json")

# ps = ParamSearch(mock=True)
#
# clf = ps.param_search(x, y)
#
# with open(params_path, "w") as outfile:
#     # writing to json file
#     json.dump(clf.best_params_, outfile)

with open(params_path, 'r') as openfile:
    # Reading from json file
    best_params = json.load(openfile)

print(best_params)

svc = SVC(**best_params)
evaluate_scores(x, y, svc, "accuracy")
