from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

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
