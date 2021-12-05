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

import Models

model_type = 'DTR' 

import Data_analys



file_name = 'dane.csv'

df = pd.read_csv(file_name) # to trzeba przenieść do Main

X_train, X_test, y_train, y_test = Data_analys.data_to_model(df)

if(len(X_train[0])>9):
    model_type = 'GA'



#hyperparameters
criterion,splitter = 'squared_error','best' #used when model_type is 'DTR'

kernel, C, epsilon = 'linear', 1, 10  #used when model_type is 'SVR'

n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = 200, 20, 2, 2, 'sqrt' #used when model_type is 'RF'

n_neighbors,metric = 5,'Euclidean' # used when model_type is 'KNN'

n=2 # Used when database has more then 10 features


if(model_type == 'DTR'):
    model = Models.DTR_model(X_train,X_test,y_train,criterion,splitter)
elif(model_type == 'SVR'):
    model = Models.SVR_model(X_train, X_test, y_train, kernel, C, epsilon)
elif(model_type == 'RF'):
    model = Models.RF_model(X_train, X_test, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
elif(model_type == 'GA'):
    model = Models.GA_model(X_train,y_train,X_test,n);


def leverage(X_test):
    X = []

    for i in range(len(X_test)):
        Y = [1]+[X_test[i][j] for j in range(len(X_test[i]))]
        X.append(Y)
    X = np.array(X)
    X = np.asmatrix(X)  
    XT = X.transpose()
    XX1 = np.linalg.inv(np.dot(XT,X))
    H = np.dot(np.dot(X,XX1),XT)
    hi = [np.asarray(H)[i][i] for i in range(len(np.asarray(H)))]
    return(hi)

def std_res(y_pred,y_test,hi):
    std_res = []
    for i in range(len(y_pred)):
        std_res.append(y_test[i]/(y_pred[i]*np.sqrt(1-hi[i])))
    return(std_res)


y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
hi= leverage(X_test)

print('ypred = ',len(y_pred),'\n ytrain = ',len(y_test),'\n hi = ',len(hi),'\n Xtest = ',len(X_test))

std_res1 = std_res(y_pred,y_test,hi)
hi_train = leverage(X_train)
std_res2 = std_res(y_pred_train,y_train,hi_train)
print('RMSE = ',mean_squared_error(y_test, y_pred,squared=False),' R^2 = ', r2_score(y_test,y_pred))


plot_lm_4 = plt.figure()
plt.scatter(hi, std_res1, alpha=0.5)
plt.scatter(hi_train, std_res2, alpha=0.5)
plot_lm_4.axes[0].set_xlim(0, max(hi)+0.01)
plot_lm_4.axes[0].set_ylim(-3, 5)
plot_lm_4.axes[0].set_title('Residuals vs Leverage')
plot_lm_4.axes[0].set_xlabel('Leverage')
plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

# Ada to twój plot \/

x = [i for i in range(len(y_pred))]
plt.scatter(x,y_pred, color = 'red')
plt.scatter(x,y_test, color = 'green')
plt.xlabel("i'th value of predicted or true data")
plt.ylabel("Value of explained variables")
plt.show()



def app(plot_1 = plot_lm_4):
	#st.write('Na początku przeanalizujmy wykres korelacji pomiędzy zmiennymi.')
	st.pyplot(plot_1)
	
