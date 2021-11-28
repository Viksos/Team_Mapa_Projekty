import streamlit as st
import os

from MultiPage import MultiPage

import First_analys,Support_Vector_Regression  #Random_Forest, Multiple_Linear_Regression, Decision_Tree_Regression, 
app = MultiPage()

st.title("Projekt zespołu MAPA")

app.add_page("Początkowa analiza",First_analys.app)
app.add_page("Support Vectior Regression", Support_Vector_Regression.app)

#app.add_page("Random Forest Regression",Random_Forest.app)
#app.add_page("Multiple Linear Regression",Multiple_Linear_Regression.app)
#app.add_page("Decision Tree Regression",Decision_Tree_Regression.app)


app.run()