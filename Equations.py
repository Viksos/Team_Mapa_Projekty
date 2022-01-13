import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import Models, Inputs
import graphviz
import sklearn
from sklearn.tree import export_graphviz
import pydot


def get_equation(model_type, model, df):

	features = list(df.columns[1:-1])

	if model_type == "MLR":
		coef = model.coef_
		inter = model.intercept_
		coef = [round(c, 2) for c in coef]
		inter = round(inter, 2)
		equation = ' + '.join([str(f) + " * " + str(c) for f, c in zip(features, coef)])
		equation = "y = " + str(equation) + " + " + str(inter)
		print(equation)
		st.latex(equation)

	elif model_type == "DTR":
		st.write("Showing only 2 levels of the tree:")
		fig = export_graphviz(model, out_file=None, max_depth=2, feature_names=features)
		st.graphviz_chart(fig)

		export_graphviz(model, out_file="tree.dot", max_depth=5, feature_names=features)
		(graph,) = pydot.graph_from_dot_file('tree.dot')
		graph.write_png('tree.png')
		with open("tree.png", "rb") as file:
			st.download_button(label="Download 5 level tree", data=file, file_name="tree.png", mime="image/png")


	elif model_type == "RF":
		st.write("Showing only the beginning of the tree:")
		fig = export_graphviz(model.estimators_[0], out_file=None, max_depth=2, feature_names=features)
		st.graphviz_chart(fig)

		export_graphviz(model.estimators_[0], out_file="tree.dot", max_depth=5, feature_names=features)
		(graph,) = pydot.graph_from_dot_file('tree.dot')
		graph.write_png('tree.png')
		with open("tree.png", "rb") as file:
			st.download_button(label="Download 5 level tree", data=file, file_name="tree.png", mime="image/png")



	elif model_type == "SVR":
		coef = model.coef_
		inter = model.intercept_
		coef = [round(c, 2) for c in coef[0]]
		inter = round(inter[0], 2)
		equation = f"Coefficients: {coef}, Intercept: {inter}"
		st.write(equation)