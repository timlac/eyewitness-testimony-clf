from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, auc, \
    classification_report
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np


from src.utils.get_splits import get_splits


def plot_prec_recall_curve(clf, x, y):
    precisions = []
    recalls = []
    threshold_indices = []

    splits = get_splits(x, y)

    for train_index, test_index in splits:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)

        y_score = clf.decision_function(X_test)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)

        mean_recall = np.linspace(0, 1, 100)
        precision_interp = np.interp(mean_recall, np.flip(recall), np.flip(precision))
        thresholds_interp = np.interp(mean_recall, np.flip(recall[:-1]), np.flip(thresholds))
        idx = getnearpos(thresholds_interp, 0)
        threshold_indices.append(idx)

        precisions.append(precision_interp)
        recalls.append(mean_recall)

    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)

    for idx in threshold_indices:
        # Highlight the threshold by plotting a vertical line
        plt.axvline(x=mean_recall[idx], color='r', linestyle='--')

    plt.step(mean_recall, mean_precision, color='b', alpha=0.2, where='post')
    plt.fill_between(mean_recall, mean_precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("2-class Precision-Recall curve with {} as positive label".format(pos_label))
    plt.show()


def getnearpos(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def plot_auc_curve(svc, x, y):
    # Compute the mean and standard deviation of the ROC AUC scores over the folds
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    splits = get_splits(x, y)
    for train, test in splits:
        # Train the classifier on the training data
        svc.fit(x[train], y[train])

        # Predict probabilities for the test data
        y_prob = svc.predict_proba(x[test])

        # Compute the ROC AUC score for this fold
        roc_auc = roc_auc_score(y[test], y_prob[:, 1])
        aucs.append(roc_auc)

        # Compute the ROC curve for this fold
        fpr, tpr, thresholds = roc_curve(y[test], y_prob[:, 1])
        tpr = np.interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    # Compute the mean and standard deviation of the ROC curves over the folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    # Plot the mean ROC curve and the shaded region representing the standard deviation
    plt.plot(mean_fpr,
             mean_tpr,
             color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (np.mean(aucs), np.std(aucs)))

    plt.fill_between(mean_fpr,
                     mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color='grey',
                     alpha=0.3,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
