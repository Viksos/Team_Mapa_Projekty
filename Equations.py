import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import Models, Inputs
import graphviz
from sklearn.tree import export_graphviz

def get_equation(model_type, model):

	if model_type == "MLR":
		coef = model.coef_
		inter = model.intercept_
		coef = [round(c, 2) for c in coef]
		inter = round(inter, 2)
		equation = f"Coefficients: {coef}, Intercept: {inter}"
		st.write(equation)

	elif model_type == "DTR":
		st.write("Showing only the beginning of the tree:")
		fig = export_graphviz(model, out_file=None, max_depth=2)
		st.graphviz_chart(fig)


	elif model_type == "RF":
		st.write("Showing only the beginning of the tree:")
		fig = export_graphviz(model.estimators_[0], out_file=None, max_depth=2)
		st.graphviz_chart(fig)

