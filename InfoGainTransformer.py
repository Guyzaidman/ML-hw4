from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest


class InfoGainTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper class for information gain function to apply to Pipeline
    """
    def __init__(self, k=5):
        self.k = k
        self.idx = None
        self.scores = None

    def fit(self, X, y):
        self.scores = mutual_info_classif(X, y)
        self.idx = np.argsort(self.scores)[::-1][:self.k]
        return self

    def transform(self, X):
        return X[:, self.idx]

    # # def fit_transform(self, X, y=None, **fit_params):
    # @staticmethod
    # def skb(X, y):
    #     scores = mutual_info_classif(X, y)
    #     return scores
    #     # idx = np.argsort(scores)[::-1][:self.k]
    #     # return X[:, idx]


if __name__ == '__main__':
    X, y = make_classification(random_state=0, n_informative=8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pipe = Pipeline([
        ('pca', PCA()),
        ('ig', InfoGainTransformer()),
    ])
    pipe2 = Pipeline([
        ('pca', PCA()),
        ('ig', InfoGainTransformer()),
        ('svc', SVC())
    ])

    pipe3 = Pipeline([
        ('first_pipe', pipe),
        ('secod_pipe', pipe2),
    ])

    pipe3.fit(X_train,y_train)
    print(pipe3.predict(X_test))
