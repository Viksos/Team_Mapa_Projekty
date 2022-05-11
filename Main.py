import streamlit as st
import os
import pyxlsb
import openpyxl

from MultiPage import MultiPage

import First_analys,Plots,Inputs, Save_to_xls #Support_Vector_Regression  #Random_Forest, Multiple_Linear_Regression, Decision_Tree_Regression,
app = MultiPage()

st.title("Projekt zespołu MAPA")

app.add_page('Inputs',Inputs.app)
app.add_page("Initial analysis",First_analys.app)
#app.add_page("Support Vectior Regression", Support_Vector_Regression.app)
app.add_page("Plots",Plots.app)
app.add_page("Save_to_xls", Save_to_xls.app)

app.run()
