
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




def data_to_model(df):

	split_size = 0.33 

	predictions = pd.DataFrame() 
	
	X, y = df.iloc[:, 1:-1].to_numpy(), df['y'].to_numpy()

	scaler = StandardScaler()

	X = scaler.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = split_size, random_state=1)

	return(X_train, X_test, y_train, y_test)


def important_value(df):
	correlation = df.iloc[:,1:len(df.columns)].corr()
	return(correlation.tail(n=1).to_numpy()[0])	


def correlation(df, color):
	fig, ax = plt.subplots()
	corr = df.corr()
	sns.heatmap(corr,annot = True, cmap = color)
	plt.title('Features Correlating ')
	return(fig)


