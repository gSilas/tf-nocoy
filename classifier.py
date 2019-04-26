import itertools
import math

import numpy as np
from sklearn import svm
from sklearn.datasets import make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.externals.joblib import Parallel, cpu_count, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

data_file = "PXD007612_clean.csv"
test_size = 0.3
cv_size = 7
n_jobs = -1

"""
list of classifiers to iterate through, format => [(func_definition, str_func_name), ...]
if str_func_name is None the class name of the classifier is taken
"""
funcs =[(LogisticRegression(solver='lbfgs', multi_class='ovr', class_weight=None, tol=1e-5, max_iter=200), None),
        (svm.SVC(gamma='scale', class_weight = None, tol=1e-5), "SVC-RBF"),
        (svm.NuSVC(gamma='scale', class_weight = None, tol=1e-5), "NuSVC-RBF"),
        (svm.SVC(gamma='scale', class_weight = None, kernel='sigmoid', tol=1e-5), "SVC-Sigmoid"),
        (GradientBoostingClassifier(n_estimators = 400, tol=1e-5, max_depth=3), "GradientBoosting400maxdepth3"),
        (GradientBoostingClassifier(n_estimators = 400, tol=1e-5, max_depth=4), "GradientBoosting400maxdepth4"),
        (GradientBoostingClassifier(n_estimators = 400, tol=1e-5, max_depth=5), "GradientBoosting400maxdepth5"),
        (GradientBoostingClassifier(n_estimators = 400, tol=1e-5, max_depth=6), "GradientBoosting400maxdepth6"),
        (GradientBoostingClassifier(n_estimators = 400, tol=1e-5, max_depth=7), "GradientBoosting400maxdepth7"),
        (GradientBoostingClassifier(n_estimators = 400, tol=1e-5, max_depth=8), "GradientBoosting400maxdepth8"),
        (GradientBoostingClassifier(n_estimators = 500, tol=1e-5, max_depth=3), "GradientBoosting500maxdepth3"),
        (GradientBoostingClassifier(n_estimators = 500, tol=1e-5, max_depth=4), "GradientBoosting500maxdepth4"),
        (GradientBoostingClassifier(n_estimators = 500, tol=1e-5, max_depth=5), "GradientBoosting500maxdepth5"),
        (GradientBoostingClassifier(n_estimators = 500, tol=1e-5, max_depth=6), "GradientBoosting500maxdepth6"),
        (GradientBoostingClassifier(n_estimators = 500, tol=1e-5, max_depth=7), "GradientBoosting500maxdepth7"),
        (GradientBoostingClassifier(n_estimators = 500, tol=1e-5, max_depth=8), "GradientBoosting500maxdepth8"),
        (GradientBoostingClassifier(n_estimators = 600, tol=1e-5, max_depth=3), "GradientBoosting600maxdepth3"),
        (GradientBoostingClassifier(n_estimators = 600, tol=1e-5, max_depth=4), "GradientBoosting600maxdepth4"),
        (GradientBoostingClassifier(n_estimators = 600, tol=1e-5, max_depth=5), "GradientBoosting600maxdepth5"),
        (GradientBoostingClassifier(n_estimators = 600, tol=1e-5, max_depth=6), "GradientBoosting600maxdepth6"),
        (GradientBoostingClassifier(n_estimators = 600, tol=1e-5, max_depth=7), "GradientBoosting600maxdepth7"),
        (GradientBoostingClassifier(n_estimators = 600, tol=1e-5, max_depth=8), "GradientBoosting600maxdepth8"),
        (GradientBoostingClassifier(n_estimators = 700, tol=1e-5, max_depth=3), "GradientBoosting700maxdepth3"),
        (GradientBoostingClassifier(n_estimators = 700, tol=1e-5, max_depth=4), "GradientBoosting700maxdepth4"),
        (GradientBoostingClassifier(n_estimators = 700, tol=1e-5, max_depth=5), "GradientBoosting700maxdepth5"),
        (GradientBoostingClassifier(n_estimators = 700, tol=1e-5, max_depth=6), "GradientBoosting700maxdepth6"),
        (GradientBoostingClassifier(n_estimators = 700, tol=1e-5, max_depth=7), "GradientBoosting700maxdepth7"),
        (GradientBoostingClassifier(n_estimators = 700, tol=1e-5, max_depth=8), "GradientBoosting700maxdepth8"),
        (ExtraTreesClassifier(n_estimators = 350), "ExtraTrees350"),
        (ExtraTreesClassifier(n_estimators = 300), "ExtraTrees300"),
        (ExtraTreesClassifier(n_estimators = 250), "ExtraTrees250"),
        (ExtraTreesClassifier(n_estimators = 200), "ExtraTrees200"),
        (ExtraTreesClassifier(n_estimators = 150), "ExtraTrees150"),
        (ExtraTreesClassifier(n_estimators = 100), "ExtraTrees100"),
        (RandomForestClassifier(n_estimators = 350), "RandomForest350"),
        (RandomForestClassifier(n_estimators = 300), "RandomForest300"),
        (RandomForestClassifier(n_estimators = 250), "RandomForest250"),
        (RandomForestClassifier(n_estimators = 200), "RandomForest200"),
        (RandomForestClassifier(n_estimators = 150), "RandomForest150"),
        (RandomForestClassifier(n_estimators = 100), "RandomForest100"),
        (AdaBoostClassifier(n_estimators = 400), None),
        (KNeighborsClassifier(n_neighbors=5), "kNN-5-uniform"),
        (KNeighborsClassifier(n_neighbors=10), "kNN-10-uniform"),
        (KNeighborsClassifier(n_neighbors=15), "kNN-15-uniform"),
        (KNeighborsClassifier(n_neighbors=20), "kNN-20-uniform"),
        (KNeighborsClassifier(n_neighbors=50), "kNN-50-uniform"),
        (KNeighborsClassifier(n_neighbors=100), "kNN-100-uniform"),
        (KNeighborsClassifier(n_neighbors=200), "kNN-200-uniform"),
        (KNeighborsClassifier(n_neighbors=5, weights='distance'), "kNN-5-distance"),
        (KNeighborsClassifier(n_neighbors=10, weights='distance'), "kNN-10-distance"),
        (KNeighborsClassifier(n_neighbors=15, weights='distance'), "kNN-15-distance"),
        (KNeighborsClassifier(n_neighbors=20, weights='distance'), "kNN-20-distance"),
        (KNeighborsClassifier(n_neighbors=50, weights='distance'), "kNN-50-distance"),
        (KNeighborsClassifier(n_neighbors=100, weights='distance'), "kNN-100-distance"),
        (KNeighborsClassifier(n_neighbors=200, weights='distance'), "kNN-200-distance"),
        (GaussianNB(), None),
        (QuadraticDiscriminantAnalysis(tol=1e-5), None)
        ]


# is normally calculated by joblib.Parallel, but we need it upfront to divide the seeds accordingly
if n_jobs < 0:
    # inverse definition e.g. -2 = all cpu's but one
    n_jobs = max(cpu_count() + 1 + n_jobs, 1)
else:
    n_jobs = n_jobs

n_jobs = min(n_jobs, cv_size)

seeds_per_job = (cv_size // n_jobs) * np.ones(n_jobs, dtype=np.int)     # partition seeds
seeds_per_job[:cv_size % n_jobs] += 1   # partition the remaining seeds over all jobs
seeds_per_job = [0] + np.cumsum(seeds_per_job).tolist()    # get the sums i.e. the actual indices

# read data and split x,y columns
my_data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
X = my_data[:,:-1]
y = my_data[:,-1]

# load a test dataset
#X,y = make_moons(noise=0.3, random_state=0)

random_state = np.random.RandomState(16510)
seeds = random_state.randint(np.iinfo(np.int32).max, size=cv_size)

def parallel_fit(func, seeds, X, y, test_size):
    scores_train = []
    scores_test = []
    precision = []
    f1 = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        try:
            func.set_params(random_state = seed)
        except Exception: 
            pass

        est = func.fit(X_train, y_train)
        y_test_pred = est.predict(X_test)
        y_train_pred = est.predict(X_train)
        scores_train.append(accuracy_score(y_train, y_train_pred))
        scores_test.append(accuracy_score(y_test, y_test_pred))
        precision.append(precision_score(y_test, y_test_pred))
        f1.append(f1_score(y_test, y_test_pred))
        
    return scores_train, scores_test, precision, f1


print('{:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<}'.format('~|Acc@Train',
                                                                                'IQR|Acc@Train', 
                                                                                '~|Acc@Test',
                                                                                'IQR|Acc@Test', 
                                                                                '~|Prec@Test',
                                                                                'IQR|Prec@Test', 
                                                                                '~|F1@Test',
                                                                                'IQR|F1@Test', 
                                                                                'Config'))
for func, funcname in funcs:
    try:
        func.set_params(n_jobs=1)
    except Exception: 
        pass

    result = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(parallel_fit)(func, 
                                        seeds[seeds_per_job[i]:seeds_per_job[i+1]], 
                                        X, 
                                        y, 
                                        test_size) for i in range(n_jobs))
    scores_train, scores_test, precision, f1 = zip(*result)
    scores_train = list(itertools.chain.from_iterable(scores_train))
    scores_test = list(itertools.chain.from_iterable(scores_test))
    precision = list(itertools.chain.from_iterable(precision))
    f1 = list(itertools.chain.from_iterable(f1))
    if funcname is None:
        funcname = str(func)
        funcname = funcname[:funcname.find('(')]
    print('{:<13.3f} {:<16.5f} {:<13.3f} {:<16.5f} {:<13.3f} {:<16.5f} {:<13.3f} {:<16.5f} {:<}'.format(np.median(scores_train), 
                                                                                                            np.subtract(*np.percentile(scores_train, [75, 25])), 
                                                                                                            np.median(scores_test), 
                                                                                                            np.subtract(*np.percentile(scores_test, [75, 25])),
                                                                                                            np.median(precision), 
                                                                                                            np.subtract(*np.percentile(precision, [75, 25])),
                                                                                                            np.median(f1), 
                                                                                                            np.subtract(*np.percentile(f1, [75, 25])), 
                                                                                                            funcname))
