from sklearn.model_selection import StratifiedKFold


def get_splits(x, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    return skf.split(x, y)
