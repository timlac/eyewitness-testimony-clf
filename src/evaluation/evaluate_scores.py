from sklearn.model_selection import cross_validate
import numpy as np
from src.utils.get_splits import get_splits


def evaluate_scores(x, y, clf, scoring_method):

    splits = get_splits(x, y)

    # get scores
    scores = cross_validate(X=x, y=y,
                            estimator=clf,
                            scoring=[scoring_method],
                            cv=splits,
                            n_jobs=-1,
                            return_train_score=True
                            )

    print('printing {} measures'.format(scoring_method))
    print('avg (train):', np.mean(scores['train_{}'.format(scoring_method)]))
    print('std (train):', np.std(scores['train_{}'.format(scoring_method)]))
    print('avg (validation):', np.mean(scores['test_{}'.format(scoring_method)]))
    print('std (validation):', np.std(scores['test_{}'.format(scoring_method)]))



