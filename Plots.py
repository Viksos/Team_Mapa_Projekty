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
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px


import Models, Inputs, Equations

import Data_analys


df = Inputs.return_df() #pd.read_csv(file_name) # to trzeba przenieść do Main
X_train, X_test, y_train, y_test = Data_analys.data_to_model(df)

if (len(X_train[0]) > 9):
	model_type = 'GA'

default_value= None


def models(model_type, var):
	if(model_type == 'DTR'):
		model = Models.DTR_model(X_train, y_train,*var)
	elif(model_type == 'SVR'):
		model = Models.SVR_model(X_train, y_train, *var)
	elif(model_type == 'RF'):
		model = Models.RF_model(X_train, y_train, *var)
	elif(model_type == 'GA'):
		model = Models.GA_model(X_train,y_train, *var)
	elif(model_type == 'MLR'):
		model = Models.MLR_model(X_train, y_train)
	elif(model_type == 'KNN'):
		model = Models.KNN_model(X_train, y_train, *var)

	return(model)


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


#Parameters for Test data
def parameters_to_plot(X_test,y_test,model):
	num_of_parameters = len(X_test[0]) # Num paramer for standars residual 

	y_pred = model.predict(X_test)
	hi= leverage(X_test)
	std_res1 = std_res(y_pred,y_test,hi,num_of_parameters)

	y_pred_train = model.predict(X_train)
	res, MSExn = MSE_calc(y_pred,y_test)
	MSE = (1/(len(y_pred)))*sum(MSExn)
	Cooks_distance = Cook_distance(res,MSE,num_of_parameters,hi)

	return(hi,std_res1,Cooks_distance,num_of_parameters,y_pred)


def Williams_plot(hi,std_res1,Cooks_distance,num_of_parameters,a,b,model_type):
	plot_lm_4 = plt.figure()
	plt.scatter(hi, std_res1, alpha=0.5)
	if(model_type == "DTR"):
		pass
	else:
		plot_lm_4.axes[0].axhline(y=0, color='grey', linestyle='dashed')
	sns.regplot(hi, std_res1,ci=False,line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

	leverage_top_3 = np.flip(np.argsort(Cooks_distance), 0)[:3]
	for i in leverage_top_3:
		plot_lm_4.axes[0].annotate(i,xy=(hi[i],std_res1[i]))


	plot_lm_4.axes[0].plot([max(hi), max(hi)], [a, b], label='h critical')

	plot_lm_4.axes[0].set_xlim(0, max(hi)+0.01)
	plot_lm_4.axes[0].set_ylim(a,b)
	plot_lm_4.axes[0].set_title('Residuals vs Leverage')
	plot_lm_4.axes[0].set_xlabel('Leverage')
	plot_lm_4.axes[0].set_ylabel('Standardized Residuals')


	graph(lambda x: np.sqrt((0.5 * num_of_parameters * (1 - x)) / x), np.linspace(0.001, max(hi), 50), 'Cook\'s distance') # 0.5 line
	graph(lambda x: np.sqrt((1 * num_of_parameters * (1 - x)) / x), np.linspace(0.001, max(hi), 50)) # 1 line
	plot_lm_4.legend(loc='upper right')

	return(plot_lm_4)


def predictions_plot(X, y, model, visuals):
	if type(X) is list:
		X_test, X_train = X
		y_test, y_train = y
		predictions_train = model.predict(X_train)
		predictions_test = model.predict(X_test)

		df_test = pd.DataFrame({'Predicted':predictions_test, 'Actual':y_test})
		df_test['Type'] = "Test"
		df_train = pd.DataFrame({'Predicted':predictions_train, 'Actual':y_train})
		df_train['Type'] = "Train"
		indexes = df[['Unnamed: 0', 'y']].rename(columns={'Unnamed: 0': 'Index'})
		plot_data = pd.concat([df_test, df_train])
		plot_data = plot_data.merge(indexes, how='left', left_on='Actual', right_on='y', copy=False).drop(['y'], axis=1)
		print(plot_data.columns)

		fig = px.scatter(plot_data,
						 x='Predicted', y='Actual',
						 title="Predicted vs Actual",
						 trendline="ols", color='Type',
						 opacity=0.6, trendline_scope = 'trace',
						 color_discrete_sequence=[visuals['points_color'], visuals['points_color2']],
						 symbol_sequence=[visuals['points_shape']],
						 hover_data=['Index'])
		fig.update_layout(font_family=visuals['font_type'], font_size=visuals['font_size'])

	else:
		predictions = model.predict(X)
		plot_data = {'Predicted':predictions, 'Actual':y}
		fig = px.scatter(plot_data,
						 x='Predicted', y='Actual',
						 title="Predicted vs Actual",
						 trendline="ols",
						 color_discrete_sequence=[visuals['points_color']],
						 symbol_sequence=[visuals['points_shape']])
		fig.update_layout(font_family=visuals['font_type'], font_size=visuals['font_size'])

	return st.plotly_chart(fig)


def print_scores(model, X_train, y_train, X_test, y_test):
	train_pred = model.predict(X_train)
	test_pred = model.predict(X_test)
	train_scores = [mean_squared_error(y_train, train_pred), r2_score(y_train, train_pred)]
	test_scores = [mean_squared_error(y_test, test_pred), r2_score(y_test, test_pred)]
	scores = pd.DataFrame([test_scores, train_scores], columns=['RMSE', 'R2 Score'], index=['Test', 'Train'])
	return scores


def app():

	model_type = Inputs.return_model_type()
	var = Inputs.return_model_var()
	model = models(model_type, var)

	st.markdown("---")
	st.header("RMSE and R2 score results")
	st.write("The model has been fitted, see the results on the test and train set below:")
	scores = print_scores(model, X_train, y_train, X_test, y_test)
	st.dataframe(scores)

	if st.checkbox("Model's equation"):
		Equations.get_equation(model_type, model, df)

	st.markdown("---")
	st.header("Results visualization on the plots")
	data_type = st.selectbox('Choose type of data to plot :',['Test','Train','Test&Train'])


	#customizing the look of plots
	if st.checkbox("Adjust the look of plots"):
		font_size = st.select_slider('Choose the font size :', [12, 18, 24, 30])
		font_type = st.selectbox('Choose the font type :', ['Consolas', 'Times New Roman', 'Calibri', 'Comic Sans MS'])
		points_shape = st.selectbox('Choose the shape of data points :', ['circle', 'diamond', 'x', 'star'])
		points_color = st.color_picker('Choose the color :')
		visuals = {'font_size':font_size, 'font_type':font_type, 'points_shape':points_shape, 'points_color':points_color}

		if data_type == 'Test&Train':
			points_color2 = st.color_picker('Choose the second color :')
			visuals['points_color2'] = points_color2
	else:
		visuals = {'font_size':18, 'font_type':'Calibri', 'points_shape':'circle', 'points_color':'#1f77b4', 'points_color2':'#d62728'}


	#predicted vs actual values plot
	st.markdown("##### Comparison of predicted value with actual values")
	if (data_type == 'Train'):
		predictions_plot(X_train, y_train, model, visuals)
	elif (data_type == 'Test'):
		predictions_plot(X_test, y_test, model, visuals)
	else:
		predictions_plot([X_test, X_train], [y_test, y_train], model, visuals)


	#Williams plot
	st.markdown("##### Williams plot")
	if(model_type == 'DTR'):
		st.write('For DTR model William\'s plot do\'t exist.')
	else:
		if(data_type == 'test'):
			hi,std_res1,Cooks_distance,num_of_parameters,y_pred = parameters_to_plot(X_test,y_test,model)

		else:
			hi,std_res1,Cooks_distance,num_of_parameters,y_pred = parameters_to_plot(X_train,y_train,model)

		a = st.number_input('Plot range from:',value = -(max(std_res1)+0.5*max(std_res1)))
		b = st.number_input("To :", value = (max(std_res1)+0.5*max(std_res1)))
		plot_1 = Williams_plot(hi,std_res1,Cooks_distance,num_of_parameters,a,b,model_type)
		plot_1.savefig('Williams_plot.')
		st.pyplot(plot_1)

		with open('Williams_plot.png', 'rb') as file:
			btn = st.download_button(
				label="Download image",
				data=file,
				file_name='Williams_plot.png',
				mime="image/png"
			)


	#st.write('RMSE = ',mean_squared_error(y_test, y_pred,squared=False),' R^2 = ', r2_score(y_test,y_pred))

