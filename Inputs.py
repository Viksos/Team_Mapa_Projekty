import streamlit as st


def app():

	model_type = st.selectbox('Choose model : ',['DTR','SVR','MLR','GA','RF'])

	if(model_type == 'DTR'):

		criterion,splitter = st.selectbox('Citerion',["squared_error", "friedman_mse", "absolute_error", "poisson"]),st.selectbox('splitter',["best","random"])

		return(criterion,splitter)

	elif(model_type == 'SVR'):

		kernel, C, epsilon = st.selectbox('Kernel',['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),st.number_input('C',value = 1),st.number_input('epsilon',value = 10)
	
		return(kernel, C, epsilon)

	elif(model_type == 'MLR'):

		n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = st.number_input('n_estimators :' ,value = 200),st.number_input('max_depth :' ,value = 20), st.number_input('min_samples_split :' ,value = 2), st.number_input('min_samples_leaf :' ,value = 2),st.selectbox('max_features',['sqrt','auto','log2'])

		return(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)

	elif(model_type == 'KNN'):
		n_neighbors,metric = st.number_input('n_neighbors',value = 5),st.text_input('Metric','Euclidean')


		return(n_neighbors,metric)

	elif(model_type == 'GA'):
		n = st.number_input('Number of parametrs',value = 2)

		return(n)
