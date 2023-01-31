from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np


from src.utils.get_splits import get_splits


def plot_prec_recall_curve(clf, x, y):
    # Use cross_val_predict to predict the scores for each sample
    y_scores = cross_val_predict(clf, x, y, cv=5, method='predict_proba')[:, 1]

    # Compute the precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, y_scores)

    # Plot the precision-recall curve
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


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
