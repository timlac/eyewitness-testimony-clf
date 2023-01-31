from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class ParamSearch:

    # regularization parameter, lower C -> more regularization,
    # large C -> less regularization
    parameters = {
        'class_weight': [None,
                         'balanced',
                         {0: 0.1, 1: .9},
                         {0: 0.3, 1: 0.7},
                         {0: 0.5, 1: 0.5},
                         {0: 0.7, 1: 0.3},
                         {0: 0.1, 1: 0.9}],

        "C": [0.1, 1, 5, 10, 25, 50, 75, 100],

        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],

        "kernel": ['rbf', 'linear', 'poly', 'sigmoid']
    }

    mock_parameters = {
        'class_weight': ['balanced'],
        "C": [1],
        "gamma": [0.1],
        "kernel": ['linear']
    }

    def __init__(self, mock=False):
        if mock:
            self.parameter_set = self.mock_parameters
        else:
            self.parameter_set = self.parameters

    def param_search(self, x, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True)

        svc = SVC()
        clf = GridSearchCV(estimator=svc,
                           param_grid=self.parameter_set,
                           scoring='roc_auc',
                           verbose=5,
                           cv=skf.split(x, y),
                           n_jobs=-1,
                           )

        clf.fit(x, y)
        print(clf.best_params_)
        return clf
