import os
import json
import numpy as np


import pandas as pd
from sklearn.svm import SVC
from src.utils.get_splits import get_splits
from src.preprocessing.normalization.functional_normalization import functional_normalize_by
from src.evaluation.evaluate_scores import evaluate_scores
from config import ROOT_DIR, AUDIO_FUNCTIONALS_EGEMAPS_COLS, AUDIO_VISUALIZATION_COLS
from src.visualization.plot_means import plot_means_and_stds
from src.visualization.confusion_matrix import ConfusionMatrixCreator, plot_conf_mat
from src.evaluation.classification_report import create_classification_report
from src.learning.parameter_tuning import ParamSearch


from sklearn import metrics
from sklearn.model_selection import cross_val_predict

from src.visualization.histogram import plot_hist

# paths
input_data = os.path.join(ROOT_DIR, "files/data/opensmile_functionals.csv")
output_data = os.path.join(ROOT_DIR, "files/out")

df = pd.read_csv(input_data, delimiter=";")

y = df["Accuracy"].values
x = df[AUDIO_FUNCTIONALS_EGEMAPS_COLS].values
participant = df["Participant"].values

plot_hist(y, "labels")


# plot_means_and_stds(x, y, AUDIO_VISUALIZATION_COLS, "pre")

x = functional_normalize_by(x, participant, method="min_max")

# plot_means_and_stds(x, y, AUDIO_VISUALIZATION_COLS, "post")

params_path = os.path.join(output_data, "best_params.json")

ps = ParamSearch(mock=True)

clf = ps.param_search(x, y, scoring="roc_auc")

with open(params_path, "w") as outfile:
    # writing to json file
    json.dump(clf.best_params_, outfile)

with open(params_path, 'r') as openfile:
    # Reading from json file
    best_params = json.load(openfile)


### Evaluation
print(best_params)
svc = SVC(**best_params)
# splits = get_splits(x, y)
# evaluate_scores(x=x,
#                 y=y,
#                 clf=svc,
#                 splits=splits,
#                 scoring_method="accuracy")

splits = get_splits(x, y)
y_pred = cross_val_predict(svc, x, y, cv=splits)

plot_hist(y_pred, "predictions")

report = metrics.classification_report(y_true=y, y_pred=y_pred)

print(report)
splits = get_splits(x, y)

conf_mat_creator = ConfusionMatrixCreator(clf=svc)
conf_mat = conf_mat_creator.calculate_avg_conf_matrix(x, y, splits)

print(conf_mat)

plot_conf_mat(conf_mat)



