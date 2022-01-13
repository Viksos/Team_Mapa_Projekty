import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from genetic_selection import GeneticSelectionCV
from sklearn.metrics import r2_score


def GA_model(X_train,y_train,n):
    estimator = linear_model.LinearRegression()

    selector = GeneticSelectionCV(estimator,cv=5,
                                      verbose=1,
                                      scoring="r2",
                                      max_features=n,
                                      n_population=50,
                                      crossover_proba=0.5,
                                      mutation_proba=0.2,
                                      n_generations=40,
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,
                                      n_gen_no_change=10,
                                      caching=True,
                                      n_jobs=-1)

    
    selector = selector.fit(X_train, y_train)
    return(selector)


#Model MLR
def MLR_model(X_train,y_train):
    MLR = LinearRegression()
    MLR = MLR.fit(X_train, y_train)
    

    return(MLR)


#Model KNN
def KNN_model(X_train,y_train, n_neighbors, metric):
    KNN = KNeighborsRegressor()
    KNN = KNN.fit(X_train, y_train)

    return(KNN)


# Model RF
def RF_model(X_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    RF = ensemble.RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features)
    RF = RF.fit(X_train, y_train)
    
    return RF


# Model SVR
def SVR_model(X_train, y_train, kernel, C, epsilon):
    SVR = svm.SVR(kernel=kernel, C=C, epsilon=epsilon)
    SVR = SVR.fit(X_train, y_train)
    return SVR

    
# Model DTR
def DTR_model(X_train,y_train,criterion1, splitter1):
    DTR = tree.DecisionTreeRegressor(criterion=criterion1,splitter= splitter1)
    DTR = DTR.fit(X_train, y_train)
    return(DTR)


