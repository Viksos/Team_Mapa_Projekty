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
import streamlit as st

import Models, Inputs

# model_type = 'MLR'

import Data_analys

file_name = 'dane.csv'

df = pd.read_csv(file_name)  # to trzeba przenieść do Main

X_train, X_test, y_train, y_test = Data_analys.data_to_model(df)


if (len(X_train[0]) > 9):
    model_type = 'GA'

# hyperparameters
criterion, splitter = 'squared_error', 'best'  # used when model_type is 'DTR'

kernel, C, epsilon = 'linear', 1, 10  # used when model_type is 'SVR'

n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = 200, 20, 2, 2, 'sqrt'  # used when model_type is 'RF'

n_neighbors, metric = 5, 'Euclidean'  # used when model_type is 'KNN'

n = 2  # Used when database has more then 10 features


# To wszystko powyżej będzie znajdowało się na pierwszej stronie


def models(model_type):
    if (model_type == 'DTR'):
        # criterion,splitter = Inputs.app
        model = Models.DTR_model(X_train, X_test, y_train, criterion, splitter)
    elif (model_type == 'SVR'):
        model = Models.SVR_model(X_train, y_train, kernel, C, epsilon)
    elif (model_type == 'RF'):
        model = Models.RF_model(X_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                                max_features)
    elif (model_type == 'GA'):
        model = Models.GA_model(X_train, y_train, X_test, n)
    elif (model_type == 'MLR'):
        model = Models.MLR_model(X_train, y_train)

    return (model)


def leverage(X_test):
    X = []

    for i in range(len(X_test)):
        Y = [1] + [X_test[i][j] for j in range(len(X_test[i]))]
        X.append(Y)
    X = np.array(X)
    X = np.asmatrix(X)
    XT = X.transpose()
    XX1 = np.linalg.inv(np.dot(XT, X))
    H = np.dot(np.dot(X, XX1), XT)
    hi = [np.asarray(H)[i][i] for i in range(len(np.asarray(H)))]
    return (hi)


def MSE_calc(y_pred, y_test):
    res = [(y_pred[i] - y_test[i]) for i in range(len(y_pred))]
    res2 = [res[i] ** 2 for i in range(len(res))]
    return (res, res2)


def std_res(y_pred, y_test, hi,
            num_of_parameters):  # Calculation based on https://en.wikipedia.org/wiki/Studentized_residual

    std_res = []

    res, res2 = MSE_calc(y_pred, y_test)

    sigma2 = (1 / (len(y_pred) - num_of_parameters)) * sum(res2)
    sigma = np.sqrt(sigma2)

    for i in range(len(y_pred)):
        if (sigma == 0):
            std_res.append(0)
        else:
            std_res.append(res[i] / (sigma * np.sqrt(1 - hi[i])))

    return (std_res)


def Cook_distance(res, MSE, num_of_coef, hi):
    D = [(res[i] ** 2 / (num_of_coef * MSE)) * (hi[i] / (1 - hi[i]) ** 2) for i in range(len(res))]

    return (D)


def graph(formula, x_range, label=None):
    """
    Helper function for plotting cook's distance lines
    """
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')


# Parameters for Test data
def parameters_to_plot(X_test, y_test, model):
    num_of_parameters = len(X_test[0])  # Num paramer for standars residual

    y_pred = model.predict(X_test)
    hi = leverage(X_test)
    std_res1 = std_res(y_pred, y_test, hi, num_of_parameters)

    y_pred_train = model.predict(X_train)
    res, MSExn = MSE_calc(y_pred, y_test)
    MSE = (1 / (len(y_pred))) * sum(MSExn)
    Cooks_distance = Cook_distance(res, MSE, num_of_parameters, hi)

    return (hi, std_res1, Cooks_distance, num_of_parameters, y_pred)


def Williams_plot(hi, std_res1, Cooks_distance, num_of_parameters, a, b, model_type):
    plot_lm_4 = plt.figure()
    plt.scatter(hi, std_res1, alpha=0.5)
    if (model_type == "DTR"):
        pass
    else:
        plot_lm_4.axes[0].axhline(y=0, color='grey', linestyle='dashed')
    sns.regplot(hi, std_res1, ci=False, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    leverage_top_3 = np.flip(np.argsort(Cooks_distance), 0)[:3]
    for i in leverage_top_3:
        plot_lm_4.axes[0].annotate(i, xy=(hi[i], std_res1[i]))

    plot_lm_4.axes[0].set_xlim(0, max(hi) + 0.01)
    plot_lm_4.axes[0].set_ylim(a, b)
    plot_lm_4.axes[0].set_title('Residuals vs Leverage')
    plot_lm_4.axes[0].set_xlabel('Leverage')
    plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

    graph(lambda x: np.sqrt((0.5 * num_of_parameters * (1 - x)) / x), np.linspace(0.001, max(hi), 50),
          'Cook\'s distance')  # 0.5 line
    graph(lambda x: np.sqrt((1 * num_of_parameters * (1 - x)) / x), np.linspace(0.001, max(hi), 50))  # 1 line
    plot_lm_4.legend(loc='upper right')

    return (plot_lm_4)


def predictions_plot(X, y, model):
    predictions = model.predict(X)
    predictions_diff = predictions - y
    predictions = pd.DataFrame([y, predictions_diff])
    predictions = predictions.T.rename(columns={0:'Actual value', 1:'Difference with predicted'})

    return st.bar_chart(predictions)


# print('RMSE = ',mean_squared_error(y_test, y_pred,squared=False),' R^2 = ', r2_score(y_test,y_pred))
def app(X_test=X_test, y_test=y_test):
    model_type = st.selectbox('Choose model : ', ['DTR', 'SVR', 'MLR', 'GA', 'RF'])
    model = models(model_type)
    data_type = st.selectbox('Choose type of data to plot :', ['Test', 'Train'])

    #predicted vs actual values plot
    st.header("Comparison of predicted value with actual values:")
    if (data_type == 'Train'):
        predictions_plot(X_train, y_train, model)
    else:
        predictions_plot(X_test, y_test, model)

    #Williams plot
    st.header("Williams plot:")
    if (model_type == 'DTR'):
        st.write('For DTR model William\'s plot do\'t exist.')
    else:
        if (data_type == 'test'):
            hi, std_res1, Cooks_distance, num_of_parameters, y_pred = parameters_to_plot(X_test, y_test, model)

        else:
            hi, std_res1, Cooks_distance, num_of_parameters, y_pred = parameters_to_plot(X_train, y_train, model)
        print('hi = ', hi, ' \n st_res', std_res1, '\n cook', Cooks_distance, '\n num of pa', num_of_parameters,
              '\n y_pred:', y_pred, '\n y_test ', y_test)
        a = st.number_input('Plot range from:', value=-(max(std_res1) + 0.5 * max(std_res1)))
        b = st.number_input("To :", value=(max(std_res1) + 0.5 * max(std_res1)))
        plot_1 = Williams_plot(hi, std_res1, Cooks_distance, num_of_parameters, a, b, model_type)
        st.pyplot(plot_1)

