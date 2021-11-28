import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle


#load data 
df = pd.read_csv("dane.csv")
X = df.iloc[:, 1:6]
y = df.iloc[:, 6:7]

#standarize the data 
scaler = StandardScaler()
X = scaler.fit_transform(X)

#split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#create plot of correlation
corr = df.iloc[:,1:7].corr()
heatmap = sns.heatmap(corr, square=True)
heatmap.get_figure().savefig("correlation_heatmap.png", dpi = 200)


#helper function for getting train and test score
def get_scores(out):
    train_score = out.score(X_train, y_train)
    test_score = out.score(X_test, y_test)
    return {'train score': train_score, 'test score':test_score}


#LinearRegression
out = LinearRegression().fit(X_train, y_train)
scores = get_scores(out)
pd.DataFrame([scores]).to_excel("linear_regression_scores.xlsx", index=False)


#SVM with hyperparameter grid search 
all_scores = pd.DataFrame()
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    for C in [0.1, 0.5, 1, 2]:
        for epsilon in [0.05, 0.1, 1, 10, 20]:
            out = SVR(kernel=kernel, C=C, epsilon=epsilon).fit(X_train, y_train.values.ravel())
            scores = get_scores(out)
            params = {'kernel':kernel, 'C':C, 'epsilon':epsilon}
            scores_params = pd.DataFrame([{**scores, **params}])
            all_scores = all_scores.append(scores_params)
all_scores.to_excel("svm_scores.xlsx", index=False)


#RandomTree with hyperparameter grid search 
all_scores = pd.DataFrame()
for max_depth in [5, 10, 20, 40]:
    for min_samples_split in [2, 4, 8, 12]:
        for min_samples_leaf in [1, 2, 5, 10]:
            out = DecisionTreeRegressor(max_depth=max_depth, 
                                        min_samples_split=min_samples_split, 
                                        min_samples_leaf=min_samples_leaf).fit(X_train, y_train.values.ravel())
            scores = get_scores(out)
            params = {'max_depth':max_depth, 
                      'min_samples_split':min_samples_split, 
                      'min_samples_leaf':min_samples_leaf}
            scores_params = pd.DataFrame([{**scores, **params}])
            all_scores = all_scores.append(scores_params)
all_scores.to_excel("decision_tree_scores.xlsx", index=False)


#Random Forest with hyperparameter grid search 
all_scores = pd.DataFrame()
for n_estimators in [100, 200, 300]:
    for max_depth in [10, 20, 40]:
        for min_samples_split in [2, 4]:
            for min_samples_leaf in [1, 2]:
                for max_features in ['auto', 'sqrt', 'log2']:
                    out = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                                min_samples_split=min_samples_split, 
                                                min_samples_leaf=min_samples_leaf, 
                                                max_features=max_features).fit(X_train, y_train.values.ravel())
                    scores = get_scores(out)
                    params = {'n_estimators':n_estimators,
                              'max_depth':max_depth, 
                              'min_samples_split':min_samples_split, 
                              'min_samples_leaf':min_samples_leaf, 
                              'max_features':max_features}
                    scores_params = pd.DataFrame([{**scores, **params}])
                    all_scores = all_scores.append(scores_params)
all_scores.to_excel("random_forest_scores.xlsx", index=False)


#create plot of final scores
final_df = pd.read_excel("final.xlsx")
barplt = sns.barplot(data=final_df, x='type', y='score', hue='algorithm', palette="rocket")
plt.legend(loc = 'lower right')
barplt.get_figure().savefig("final_scores_plot.png", dpi = 200)


#pickle the best model
best_model = SVR(kernel='linear', C=1, epsilon=10).fit(X_train, y_train.values.ravel())
pickle.dump(best_model, open('final_model_svm.pkl', 'wb'))

