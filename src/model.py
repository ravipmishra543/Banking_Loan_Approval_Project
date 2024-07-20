import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_logistic_regression(X_train, y_train):
    grid = {"C": np.logspace(0.1, 0.4, 4), "penalty": ["l1", "l2"]}
    logreg = LogisticRegression(random_state=0, solver='liblinear')
    logreg_cv = GridSearchCV(logreg, grid, cv=10, scoring="accuracy", n_jobs=6, verbose=1)
    logreg_cv.fit(X_train, y_train)
    return logreg_cv

def train_random_forest(X_train, y_train):
    rf_test = {"max_depth": [3, 4, 5], "n_estimators": [100, 300, 500, 600], "min_samples_leaf": [10, 15]}
    tuning = GridSearchCV(RandomForestClassifier(), param_grid=rf_test, scoring='accuracy', n_jobs=6)
    tuning.fit(X_train, y_train)
    return tuning

def train_decision_tree(X_train, y_train):
    tree_param = {'criterion': ['entropy', 'gini'], 'max_depth': list(range(2, 16)), "min_samples_leaf": list(range(1, 80, 10))}
    gsDTC = GridSearchCV(DecisionTreeClassifier(), param_grid=tree_param, cv=5, scoring="accuracy", n_jobs=6, verbose=1)
    gsDTC.fit(X_train, y_train)
    return gsDTC

def train_knn(X_train, y_train):
    param_grid_knn = {"n_neighbors": list(range(1, 200))}
    gs_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, verbose=1, n_jobs=6, scoring="accuracy")
    gs_knn.fit(X_train, y_train)
    return gs_knn
