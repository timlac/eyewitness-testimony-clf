from sklearn.model_selection import cross_val_predict
from sklearn import metrics


def create_classification_report(clf, x, y, splits):
    y_pred = cross_val_predict(clf, x, y, cv=splits)
    report = metrics.classification_report(y_true=y, y_pred=y_pred)

    print(report)

    # with open(self.classification_report_save_path + self.save_as + ".json", 'w') as fp:
    #     json.dump(report, fp)