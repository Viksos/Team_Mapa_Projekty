import streamlit as st
import os

from MultiPage import MultiPage

import First_analys,Plots #Support_Vector_Regression  #Random_Forest, Multiple_Linear_Regression, Decision_Tree_Regression, 
app = MultiPage()

st.title("Projekt zespołu MAPA")

app.add_page("Początkowa analiza",First_analys.app)
#app.add_page("Support Vectior Regression", Support_Vector_Regression.app)
app.add_page("Wykrsy",Plots.app)



app.run()