from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
import numpy as np


def evaluate_scores(x_, y_, svc, scoring_method):
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # get scores
    scores = cross_validate(X=x_, y=y_,
                            estimator=svc,
                            scoring=[scoring_method],
                            verbose=1,
                            cv=skf.split(x_, y_),
                            n_jobs=-1,
                            return_train_score=True
                            )

    print('printing {} measures'.format(scoring_method))
    print('avg (train):', np.mean(scores['train_{}'.format(scoring_method)]))
    print('std (train):', np.std(scores['train_{}'.format(scoring_method)]))
    print('avg (validation):', np.mean(scores['test_{}'.format(scoring_method)]))
    print('std (validation):', np.std(scores['test_{}'.format(scoring_method)]))