from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from enum import Enum


class Method(str, Enum):
    min_max = "min_max"
    standard = "standard"


def functional_normalize_by(x, identifiers, method):
    """
    :param x: a matrix with shape (observations, features)
    :param identifiers: np Array
    :param method: cls Method
    :return: scaled x matrix
    """
    for identifier in np.unique(identifiers):
        if method == Method.min_max:
            scaler = MinMaxScaler()
        elif method == Method.standard:
            scaler = StandardScaler()
        else:
            raise RuntimeError("Something went wrong, no scaling method chosen")

        rows = np.where(identifiers == identifier)
        x[rows] = scaler.fit_transform(x[rows])
    return x


