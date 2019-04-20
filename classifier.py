import math

import numpy as np
from sklearn import svm
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import ShuffleSplit, train_test_split


data_file = "PXD007612_clean.csv"
test_size = 0.35
cv_size = 9


LogiReg = LogisticRegression(solver='lbfgs', multi_class='ovr', class_weight=None, tol=1e-4)
SuVeMa = svm.SVC(gamma='scale', class_weight = None)
GradBoo = GradientBoostingClassifier(n_estimators = 600)
ExtTree = ExtraTreesClassifier(n_estimators = 600)
RanFor = RandomForestClassifier(n_estimators = 600)
AdaBoo = AdaBoostClassifier(n_estimators = 600)

# running all of them would take at least 1 day
funcs = [RanFor, AdaBoo, ExtTree, GradBoo, SuVeMa, LogiReg]
funcs = [GradBoo]

print('{:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<}'.format('~|Acc@Train',
                                                                                'IQR|Acc@Train', 
                                                                                '~|Acc@Test',
                                                                                'IQR|Acc@Test', 
                                                                                '~|Prec@Test',
                                                                                'IQR|Prec@Test', 
                                                                                '~|F1@Test',
                                                                                'IQR|F1@Test', 
                                                                                'Config'))
for func in funcs:
    # read data and split x,y columns
    my_data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
    X = my_data[:,:-1]
    y = my_data[:,-1]

    random_state = np.random.RandomState(16510)
    seeds = random_state.randint(np.iinfo(np.int32).max, size=cv_size)
    scores_train = []
    scores_test = []
    precision = []
    f1 = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        func.set_params(random_state = seed)
        est = func.fit(X_train, y_train)
        y_test_pred = est.predict(X_test)
        y_train_pred = est.predict(X_train)

        scores_test.append(accuracy_score(y_test, y_test_pred))
        scores_train.append(accuracy_score(y_train, y_train_pred))
        precision.append(precision_score(y_train, y_train_pred))
        f1.append(f1_score(y_train, y_train_pred))

    print('{:<13.3f} {:<16.5f} {:<13.3f} {:<16.5f} {:<13.3f} {:<16.5f} {:<13.3f} {:<16.5f} {:<}'.format(np.median(scores_train), 
                                                                                                            np.subtract(*np.percentile(scores_train, [75, 25])), 
                                                                                                            np.median(scores_test), 
                                                                                                            np.subtract(*np.percentile(scores_test, [75, 25])),
                                                                                                            np.median(precision), 
                                                                                                            np.subtract(*np.percentile(precision, [75, 25])),
                                                                                                            np.median(f1), 
                                                                                                            np.subtract(*np.percentile(f1, [75, 25])), 
                                                                                                            str(func)))


'''
~|Acc@Train IQR|Acc@Train  ~|Acc@Test  IQR|Acc@Test   Config                        
0.806       0.00091        0.733       0.00176        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                                                            decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                                                            max_iter=-1, probability=False, random_state=1859837226, shrinking=True,
                                                            tol=0.001, verbose=False)

0.691       0.00154        0.691       0.00360        LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                                                            intercept_scaling=1, max_iter=100, multi_class='ovr',
                                                                            n_jobs=None, penalty='l2', random_state=1859837226,
                                                                            solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

0.926       0.00124        0.913       0.00146        GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                                                                    learning_rate=0.1, loss='deviance', max_depth=3,
                                                                                    max_features=None, max_leaf_nodes=None,
                                                                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                                                                    min_samples_leaf=1, min_samples_split=2,
                                                                                    min_weight_fraction_leaf=0.0, n_estimators=600,
                                                                                    n_iter_no_change=None, presort='auto',
                                                                                    random_state=1859837226, subsample=1.0, tol=0.0001,
                                                                                    validation_fraction=0.1, verbose=0, warm_start=False)

1.000       0.00000        0.911       0.00174        ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                                                                            max_depth=None, max_features='auto', max_leaf_nodes=None,
                                                                            min_impurity_decrease=0.0, min_impurity_split=None,
                                                                            min_samples_leaf=1, min_samples_split=2,
                                                                            min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=None,
                                                                            oob_score=False, random_state=1859837226, verbose=0,
                                                                            warm_start=False)

1.000       0.00000        0.912       0.00130        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                                                                max_depth=None, max_features='auto', max_leaf_nodes=None,
                                                                                min_impurity_decrease=0.0, min_impurity_split=None,
                                                                                min_samples_leaf=1, min_samples_split=2,
                                                                                min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=None,
                                                                                oob_score=False, random_state=1859837226, verbose=0,
                                                                                warm_start=False)
            
0.906       0.00049        0.903       0.00250        AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                                                                            learning_rate=1.0, n_estimators=600, random_state=1859837226)                                                                            
'''