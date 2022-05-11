import streamlit as st
import Data_analys
import pandas as pd 

file_name = 'dane2.csv'

df ,model_var, model_type = pd.read_csv(file_name),['squared_error','best'],0#,'DTR'


def app():

	global df
	global model_var
	global model_type

	st.markdown("---")
	st.header("Choose Data")
	st.write(text0)

	uploaded_file = st.file_uploader("Choose a file")

	if uploaded_file is not None:
		df = pd.read_csv(uploaded_file)
	else:
		file_name = 'dane.csv'
		df = pd.read_csv(file_name)

	st.markdown("---")
	st.header("Choose Model")
	st.markdown(text1)
	model_type = st.selectbox('Choose model : ',['DTR','KNN','SVR','MLR','GA','RF'])

	st.markdown("---")
	st.header("Choose Hyperparameters")
	st.write(text2)

	if(model_type == 'DTR'):
		criterion,splitter = st.selectbox('criterion',["squared_error", "friedman_mse", "absolute_error", "poisson"]),st.selectbox('splitter',["best","random"])
		model_var = [criterion,splitter]
		st.write(dtr_text)

	elif(model_type == 'SVR'):
		kernel, C, epsilon = st.selectbox('kernel',['linear', 'poly', 'rbf', 'sigmoid']),st.number_input('C',value = 1),st.number_input('epsilon',value = 10)
		model_var = [kernel, C, epsilon]
		st.write(svr_text)

	elif(model_type == 'MLR'):
		st.write(mlr_text)

	elif(model_type == 'KNN'):
		n_neighbors,metric = st.number_input('n_neighbors',value = 5),st.text_input('metric','Euclidean')
		model_var = [n_neighbors,metric]
		st.write(knn_text)

	elif (model_type == 'RF'):
		n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = st.number_input('n_estimators :', value=200), \
																					 st.number_input('max_depth :', value=20), \
																					 st.number_input('min_samples_split :', value=2), \
																					 st.number_input('min_samples_leaf :', value=2), \
																					 st.selectbox('max_features', ['sqrt', 'auto', 'log2'])
		model_var = [n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features]
		st.write(rf_text)

	elif(model_type == 'GA'):
		n = st.number_input('number of parameters',value = 2)
		model_var = [n]
		st.write(ga_text)


def return_df():
	return(df)
def return_model_type():
	return(model_type)
def return_model_var():
	return(model_var)

text0 = """
Upload the data of your choice or leave blank to load example data. 
First column should contain the index, the middle columns should contain variables and the last column - the output. 
"""

text1 = """
Choose one of the below models to fit to the data: 

1. 'MLR' - for Multi Linear Regression
2. 'KNN' - for K-Nearest Neighbors Regression
3. 'RF' - for Random Forest Regression
4. 'SVR' - for Support Vector Regression 
5. 'DTR' - for Decision Tree Regression
"""

text2 = """
You can tweak the following parameters: 
"""

dtr_text = """>More details about available DTR hyperparameters:
- **criterion**: {“squared_error”, “friedman_mse”, “absolute_error”, “poisson”} - The function to measure the quality of a split. Supported criteria are “squared_error” for the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node, “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential splits, “absolute_error” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node, and “poisson” which uses reduction in Poisson deviance to find splits.
- **splitter**: {“best”, “random”} - the strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
"""

svr_text = """>More details about available SVR hyperparameters:
- **kernel**: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} - Specifies the kernel type to be used in the algorithm.
- **C**: float - Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
- **epsilon**: float - Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value."""

mlr_text = """There is no hyperparameters for Linear model"""

knn_text = """>More details about available KNN hyperparameters:
- **n_neighbors** : Number of neighbors to use by default for kneighbors queries.
- **metric** : The distance metric to use for the tree. It is a measure of the true straight line distance between two points in Euclidean space."""

rf_text = """>More details about available RF hyperparameters:
- **n_estimators** : int - The number of trees in the forest.
- **max_depth** : int - The maximum depth of the tree. 
- **min_samples_split** : int - The minimum number of samples required to split an internal node.
- **min_samples_leaf** : int - The minimum number of samples required to be at a leaf node. 
- **max_features** : {“auto”, “sqrt”, “log2”} - The number of features to consider when looking for the best split. If “auto”, then max_features=n_features. If “sqrt”, then max_features=sqrt(n_features). If “log2”, then max_features=log2(n_features)."""

ga_text = """>More details about available GA hyperparameters:
- **n** : int - number of columns that's should stay after GA."""
