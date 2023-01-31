import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt


class ConfusionMatrixCreator:

    def __init__(self, clf):
        self.clf = clf

    def calculate_avg_conf_matrix(self, x, y, splits):
        classes = np.unique(y)
        n_classes = len(classes)

        cum_conf_mat = np.zeros([n_classes, n_classes])

        n_groups = 0
        for train_idx, val_idx in splits:
            n_groups += 1
            x_train, x_val, y_train, y_val = x[train_idx], x[val_idx], y[train_idx], y[val_idx]

            self.clf.fit(x_train, y_train)
            y_pred = self.clf.predict(x_val)
            conf_mat = confusion_matrix(y_val,
                                        y_pred,
                                        labels=classes,
                                        normalize='true'
                                        )

            cum_conf_mat += conf_mat

        avg_conf_mat = cum_conf_mat / n_groups
        return avg_conf_mat


def plot_conf_mat(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.subplots_adjust(bottom=.25, left=.25)

    plt.show()
