import streamlit as st 
import pickle
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
df = pd.read_csv('dane.csv', ',').iloc[:,1:]
model = pickle.load(open('final_model_svm.pkl', 'rb'))



def app():

	st.write('Predykcja za pomocą modelu opartego na SVR.')
	statment = st.selectbox('Za pomocą czego chcesz wprowadzić zmienne :',['Suwaków','Wpisywania danych']) 
	if(statment == 'Suwaków'):
		desc1 = st.slider("Wartość Desc1",min_value = -3.0,max_value = 3.0)
		desc2 = st.slider("Wartość Desc2",min_value = -3.0,max_value = 3.0)
		desc3 = st.slider("Wartość Desc3",min_value = -3.0,max_value = 3.0)
		desc4 = st.slider("Wartość Desc4",min_value = -3.0,max_value = 3.0)
		desc5 = st.slider("Wartość Desc5",min_value = -3.0,max_value = 3.0)
	else:
		desc1 = st.number_input("Wartość Desc1")
		desc2 = st.number_input("Wartość Desc2")
		desc3 = st.number_input("Wartość Desc3")
		desc4 = st.number_input("Wartość Desc4")
		desc5 = st.number_input("Wartość Desc5")
	x=np.array([desc1,desc2,desc3,desc4,desc5])
	x = x.reshape(-1,5)
	y = model.predict(x)
	if st.button("Predict"):
		st.write('Przewidywana wartość y to ',y[0])


	st.write('Poniższy wykres przedstawia predykcje zmiennej y dla modelu Support Vector Regression, w którym predyktorami jest 5 zmiennych. Po wybraniu jednego predyktora oraz ustaleniu jego wartości, pozostałe zmienne przyjmowane są jako wartości stałe.')
	zmienne = ['desc1','desc2','desc3','desc4','desc5']
	zmienna = st.selectbox('Zmienna do zmiany :',zmienne)

	
	b = st.slider("Przedział dla "+zmienna,min_value=-2.9,max_value = 3.0)
	x= np.arange(-3.0,b,0.01)

	zmienne = [zmienne[i] for i in range(len(zmienne)) if(zmienne[i]!=zmienna)]
	zmienne = [eval(zmienne[0]),eval(zmienne[1]),eval(zmienne[2]),eval(zmienne[3])]
	x_pred = [np.array([x[i],zmienne[0],zmienne[1],zmienne[2],zmienne[3]]).reshape(-1,5) for i in range(len(x))]
	x_pred = np.array(x_pred)
	y = np.array([model.predict(x_pred[i]) for i in range(len(x_pred))])

	
	fig, ax = plt.subplots()
	ax.scatter(x,y)
	plt.xlabel(zmienna)
	plt.title('y('+zmienna+')')
	st.pyplot(fig)
	if st.button('Balony ?') :
		st.balloons()