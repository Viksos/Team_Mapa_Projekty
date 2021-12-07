import streamlit as st
import Data_analys
import pandas as pd 

file_name = 'dane.csv'

df ,model_var, model_type = pd.read_csv(file_name),['squared_error','best'],0#,'DTR'


def app():

	global df
	global model_var
	global model_type

	uploaded_file = st.file_uploader("Choose a file")
	if uploaded_file is not None:
	
		df=pd.read_csv(uploaded_file)
	else:
		file_name = 'dane.csv'

		df = pd.read_csv(file_name)


	fig = Data_analys.correlation(df)
	st.pyplot(fig)

	st.write(Data_analys.important_value(df))

	model_type = st.selectbox('Choose model : ',['DTR','SVR','MLR','GA','RF'])



	if(model_type == 'DTR'):

		criterion,splitter = st.selectbox('Citerion',["squared_error", "friedman_mse", "absolute_error", "poisson"]),st.selectbox('splitter',["best","random"])

		model_var = [criterion,splitter]

	elif(model_type == 'SVR'):

		kernel, C, epsilon = st.selectbox('Kernel',['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),st.number_input('C',value = 1),st.number_input('epsilon',value = 10)
	
		model_var = [kernel, C, epsilon]

	elif(model_type == 'MLR'):

		n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = st.number_input('n_estimators :' ,value = 200),st.number_input('max_depth :' ,value = 20), st.number_input('min_samples_split :' ,value = 2), st.number_input('min_samples_leaf :' ,value = 2),st.selectbox('max_features',['sqrt','auto','log2'])

		model_var = [n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features]

	elif(model_type == 'KNN'):
		n_neighbors,metric = st.number_input('n_neighbors',value = 5),st.text_input('Metric','Euclidean')


		model_var = [n_neighbors,metric]

	elif(model_type == 'GA'):
		n = st.number_input('Number of parametrs',value = 2)

		model_var = [n]



def return_df():
	return(df)
def return_model_type():
	return(model_type)
def return_model_var():
	return(model_var)
