import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.svm import SVC

from src.preprocessing.normalization.functional_normalization import functional_normalize_by
from config import ROOT_DIR, AUDIO_FUNCTIONALS_EGEMAPS_COLS, AUDIO_VISUALIZATION_COLS

def getnearpos(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

pos_label = 1
input_data = os.path.join(ROOT_DIR, "files/data/opensmile_functionals.csv")
output_data = os.path.join(ROOT_DIR, "files/out")

df = pd.read_csv(input_data, delimiter=";")

y = df["Accuracy"].values
x = df[AUDIO_FUNCTIONALS_EGEMAPS_COLS].values
participant = df["Participant"].values

x = functional_normalize_by(x, participant, method="min_max")
params_path = os.path.join(output_data, "best_params.json")

with open(params_path, 'r') as openfile:
    # Reading from json file
    best_params = json.load(openfile)

clf = SVC(**best_params, probability=True)

X_train, X_test, y_train, y_test = train_test_split(x, y)

clf.fit(X_train, y_train)


y_score = clf.decision_function(X_test)


# Use a custom threshold to make binary class predictions
# custom_threshold = 0.5
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

precision, recall, thresholds = precision_recall_curve(y_test, y_score, pos_label=pos_label)

plt.plot(thresholds, precision[:thresholds.shape[0]])
plt.title("precision and thresholds")
plt.show()

plt.plot(thresholds, recall[:thresholds.shape[0]])
plt.title("recall and thresholds")
plt.show()

idx = getnearpos(thresholds, 0)

print(idx)

print("\nrecall")
print(recall[idx-2:idx+2])
print("precision")
print(precision[idx-2:idx+2])


plt.plot(recall, precision)
plt.title("precision and recall")
# Highlight the threshold by plotting a vertical line
plt.axvline(x=recall[idx], color='r', linestyle='--')
plt.show()