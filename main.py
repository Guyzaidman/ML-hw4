from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.utils import check_X_y
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import warnings
import time
from typing import List

# --- estimators ---
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- data cleaning & preprocessing ---
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, PowerTransformer

# --- feature selection ---
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import SelectFdr, RFE
import mrmr

# --- fold strategies ---
from sklearn.model_selection import KFold, LeavePOut, LeaveOneOut
from sklearn.model_selection import GridSearchCV

# --- metrics ---
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, matthews_corrcoef

from InfoGainTransformer import InfoGainTransformer
from ClfSwitcher import ClfSwitcher


def micro_precision_recall_auc(y_test, y_proba):
    n_classes = y_proba.shape[1]
    y_test = label_binarize(y_test, classes=range(n_classes))
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_proba[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_proba[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_proba.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_proba, average="micro")
    return average_precision["micro"]


def micro_roc_auc(y_test, y_proba):
    n_classes = y_proba.shape[1]
    y_test = label_binarize(y_test, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc["micro"]


def find_fs_name(i):
    if i == 0:
        return 'pca_ig'
    elif i == 1:
        return 'f_classif'
    elif i == 2:
        return 'SelectFdr'
    elif i == 3:
        return 'RFE'
    else:  # i == 4
        return 'Mrmr'


def find_cv(X):
    """
    Find the correct cross-validation method with respect to the number of samples in the data set
    :param X: np.array - the data set
    :return: cross validation method to be executed
    """
    n_samples = X.shape[0]
    if n_samples <= 50:
        return LeavePOut(2)
    elif 50 < n_samples <= 100:
        return LeaveOneOut()
    elif 100 < n_samples <= 1000:
        return KFold(n_splits=10, shuffle=True)
    else:  # n_samples > 1000
        return KFold(n_splits=5, shuffle=True)


def calc(dataset_name: str, X: np.array, y: np.array, clean_pipe: Pipeline, fs_pipes: List[Pipeline]) -> pd.DataFrame:
    """
    Evaluate every possible combination of k features, FS algorithm, classifier and evaluation metric
    """
    cols = ['Dataset Name', 'Numer of samples', 'original number of features', 'Filtering algorithm',
            'Learning algorithm', 'Number of features selected', 'CV method', 'Fold', 'Measure type', 'Measure value',
            'Feature selection time (ms)', 'Estimator fit time (ms)', 'Estimator average predict time (ms)']
    table = pd.DataFrame(columns=cols)

    cv = find_cv(X)
    print(cv)
    for fold_idx, (train_index, test_index) in enumerate(cv.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = clean_pipe.fit_transform(X_train)
        X_test = clean_pipe.transform(X_test)

        k_features = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
        estimators = [GaussianNB(), SVC(probability=True), LogisticRegression(),
                      RandomForestClassifier(), KNeighborsClassifier()]

        for k in tqdm(k_features):
            for i, fs_pipe in enumerate(fs_pipes):
                fs_name = find_fs_name(i)
                params = {'last__k': k}
                if i == 3:
                    params = {'last__n_features_to_select': k}
                    continue
                fs_pipe.set_params(**params)

                t = time.time()
                X_train_temp = fs_pipe.fit_transform(X_train, y_train)
                fs_t = time.time() - t
                X_test_temp = fs_pipe.transform(X_test)

                for estimator in estimators:
                    t = time.time()
                    estimator.fit(X_train_temp, y_train)
                    est_fit_t = time.time() - t

                    t = time.time()
                    y_pred = estimator.predict(X_test_temp)
                    est_pred_t = (time.time() - t) / len(X_test_temp)
                    y_proba = estimator.predict_proba(X_test_temp)

                    metrics = [roc_auc_score, average_precision_score, accuracy_score, matthews_corrcoef]
                    for f_idx, F in enumerate(metrics):
                        try:
                            if y_proba.shape[1] > 2 and f_idx == 0:
                                score = micro_roc_auc(y_test, y_proba)
                            elif y_proba.shape[1] > 2 and f_idx == 1:
                                score = micro_precision_recall_auc(y_test, y_proba)
                            else:
                                score = F(y_test, y_pred)
                        except ValueError as e:
                            # print(e)
                            score = None

                        a = [dataset_name, X.shape[0], X.shape[1], fs_name, estimator.__class__.__name__,
                             k, cv, fold_idx + 1, F.__name__, score, fs_t, est_fit_t, est_pred_t]
                        table.loc[len(table)] = a

    return table


def mrmr_wrapper(X, y):
    """
    Wraps the mrmr function for SelectKBest use
    """
    pd_X, pd_y = pd.DataFrame(X), pd.Series(y)
    _, sf_scores, _ = mrmr.mrmr_classif(pd_X, pd_y, K=10, return_scores=True, show_progress=False)
    return sf_scores


def final(name, X, y):
    print(name)
    c_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('var_thresh', VarianceThreshold(threshold=0)),
        ('standard', StandardScaler(with_std=False)),
        ('guasian', PowerTransformer()),
    ])
    feature_selection_pipes = [
        Pipeline([
            ('pca', PCA()),
            ('last', InfoGainTransformer()),
        ]),

        Pipeline([
            ('last', SelectKBest(f_classif))
        ]),

        Pipeline([
            ('fdr', SelectFdr(alpha=0.1)),
            ('last', SelectKBest(f_classif))
        ]),

        Pipeline([
            ('last', RFE(SVC(kernel='linear', C=1)))
        ]),

        # Pipeline([('', None)]),

        Pipeline([
            ('last', SelectKBest(mrmr_wrapper))
        ]),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = calc(name, X, y, c_pipe, feature_selection_pipes)
        a.to_csv(name + '.csv')


if __name__ == '__main__':
    import scipy.io

    mat = scipy.io.loadmat(
        r"C:\Guy\University\4th year\semester B\Applied machine learning\hw4\DataSets\opt3 - scikit-feature\charliec443 scikit-feature master skfeature-data\lung.mat")
    X, y = mat['X'], np.ravel(mat['Y'])
    y = y - 1
    check_X_y(X, y)
    final('scikit-feature-lung', X, y)


    mat = scipy.io.loadmat(
        r"C:\Guy\University\4th year\semester B\Applied machine learning\hw4\DataSets\opt3 - scikit-feature\charliec443 scikit-feature master skfeature-data\Yale.mat")
    X, y = mat['X'], np.ravel(mat['Y'])
    y = y - 1
    check_X_y(X, y)

    final('scikit-feature-Yale', X, y)

    df = pd.read_csv(r"C:\Guy\University\4th year\semester B\Applied machine learning\hw4\DataSets\opt1 - bioconductor\breastCancerVDX.csv",index_col=0)
    df = df.T
    X, y = df.drop(columns=['oestrogenreceptorsClass']), df['oestrogenreceptorsClass']
    X, y = np.array(X), np.array(y)

    final('bioconductor-breastCancerVDX', X, y)

    df = pd.read_csv(r"C:\Guy\University\4th year\semester B\Applied machine learning\hw4\DataSets\opt1 - bioconductor\bcellViper.csv", index_col=0)
    df = df.T
    X, y = df.drop(columns=['TypeClass']), df['TypeClass']
    X, y = np.array(X), np.array(y)
    y = y - 1
    check_X_y(X, y)

    final('bioconductor-bcellViper', X, y)

