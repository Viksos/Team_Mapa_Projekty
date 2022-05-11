import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import Data_analys
import Inputs
import plotly.figure_factory as ff
import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def app():

	#st.set_option('deprecation.showPyplotGlobalUse', False)
	#df = pd.read_csv('dane.csv', ',').iloc[:,1:]

	#spr_df = Cl.Cleaner(df)




	df = Inputs.return_df().iloc[:,1:]

	st.write('Data loaded:')
	st.dataframe(df)
	st.markdown("---")

	st.write('General statistics about the data')
	st.write(df.describe())
	st.markdown("---")

	st.write('Data distribution')
	df_dist = df.iloc[:,:-1]
	fig = ff.create_distplot([df_dist[c] for c in df_dist.columns], df_dist.columns)
	st.plotly_chart(fig)
	st.markdown("---")

	st.write('Correlation between variables')
	color = st.selectbox('Choose color : ', ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds'])
	fig = Data_analys.correlation(df, color)
	st.pyplot(fig)
	st.write(Data_analys.important_value(df))


	st.markdown("---")
	st.write('Bootstrapped mean and standard deviation')
	result = pd.DataFrame({'Mean': [], 'Bootstrapped mean': [], 'Lower mean':[], 'Upper mean':[],
						   'Std': [], 'Bootstrapped std': [], 'Lower std':[], 'Upper std':[]})
	for col in df.columns[2:]:
		col = np.array(df[col])
		col_mean = col.mean()
		col_bs_mean = bs.bootstrap(col, stat_func=bs_stats.mean).value
		col_bs_mean_up = bs.bootstrap(col, stat_func=bs_stats.mean).upper_bound
		col_bs_mean_low = bs.bootstrap(col, stat_func=bs_stats.mean).lower_bound
		col_std = col.std()
		col_bs_std = bs.bootstrap(col, stat_func=bs_stats.std).value
		col_bs_std_up = bs.bootstrap(col, stat_func=bs_stats.std).upper_bound
		col_bs_std_low = bs.bootstrap(col, stat_func=bs_stats.std).lower_bound
		col_result = pd.Series([col_mean, col_bs_mean, col_bs_mean_low, col_bs_mean_up,
								col_std, col_bs_std, col_bs_std_low, col_bs_std_up], index=result.columns)
		result = result.append(col_result, ignore_index=True)
	current_index = [i for i in result.index]
	result = result.rename(index=dict(zip(current_index, df.columns[1:])))
	st.write(result)
	st.markdown("---")


	st.write('Duplex')

	def d(lst_1: list, lst_2: list):
		'''
		Return value of euclidean distance(ED)

		@param lst_1: list of first vector
		@param lst_2: list of second vector
		'''
		lst_1 = np.array(lst_1)
		lst_2 = np.array(lst_2)
		lst_1 = lst_1[~np.isnan(lst_1)]
		lst_2 = lst_2[~np.isnan(lst_2)]
		print('lst1', lst_1)
		print('lst2', lst_2)
		print('diff', lst_1 - lst_2)
		print('sum', sum((np.array(lst_1) - np.array(lst_2)) ** 2))
		return (np.sqrt(sum((np.array(lst_1) -
							 np.array(lst_2)) ** 2)))

	def duplex(df):
		'''
		Return two data frame that are split based on duplex algorythm
		@param: df where all values are numbers
		'''
		lst = df.values.tolist()
		LHS = []
		RHS = []
		#print('lst:', lst)
		for i in lst:
			ed = 0
			for j in lst:
				#print('ed:', ed)
				#print('j:', j)
				#print('i:', i)
				#print('d(i, j:', d(i, j))
				if (ed < d(i, j)):  # to find greatest ED
					ed = d(i, j)

					nd = j
			LHS.append(i)
			RHS.append(nd)
			lst.remove(i)
			lst.remove(j)
		df_1 = pd.DataFrame(LHS, columns=df.columns)
		df_2 = pd.DataFrame(RHS, columns=df.columns)
		return ([df_1, df_2])

	le = LabelEncoder()
	df = pd.read_excel('Desc_Matrix.xlsx', skiprows=[0])
	df['NP'] = le.fit_transform(df['NP'])
	df1, df2 = duplex(df)
	st.write(df1)
	st.write(df2)


'''
	st.image('correlation_heatmap.png')
	st.write('Z powyższego wykresu możemy wywnioskować, że w naszym zbiorze istnieją dane, które są ze sobą skorelowane. Największy współczynnik korelacji występuje pomiędzy zmiennymi desc5 i y. Jego wartość należy do przedziału (0.8 - 1.0), co oznacza, że istnieje znaczna zależność pomiędzy tymi dwoma zmiennymi. Wzrostowi lub spadkowi wartości jednej z tych zmiennych towarzyszy odpowiednio wzrost lub spadek wartości drugiej zmiennej. Korelacja ze zmienną y widoczna jest również w przypadku zmiennych: desc1, desc2 oraz desc3. Pozostałe dane, które należą do zbioru, posiadają współczynnik korelacji bliski lub równy 0. ')

	st.write('Poniżej przedstawione zostały wykresy, dzięki którym możemy bliżej zapoznać się ze zmiennymi, które należą do naszego zbioru danych.')
	st.pyplot(spr_df.Describe_plot(to_drop = 'y')[0])

	st.write('Na powyższym wykresie przedstawiono minimalne wartości każdej zmiennej. Możemy zauważyć, że wartości te nie różnią się od siebie w znaczący sposób. Najmniejszą z wartości przyjmuje zmienna desc5.')
	st.pyplot(spr_df.Describe_plot(to_drop = 'y')[1])

	st.write('Powyższy wykres przedstawia maksymalne wartości jakie może przyjąć każda ze zmiennych. Podobnie jak w przypadku wartości minimalnych, tutaj również możemy zauważyć nieznaczne różnice jakie występują pomiędzy danymi. Największą z możliwych wartości przyjmuje zmienna desc2.')
	st.pyplot(spr_df.Describe_plot(to_drop = 'y')[2])

	st.write('Na powyższym wykresie przedstawione zostały wartości średnie, jakie przyjmuje każda ze zmiennych. Możemy zauważyć, że jedynie zmienna desc3 posiada średnią mniejszą od 0. Największe wartości przyjmuje zmienna desc4. Możemy jednak zauważyć, że przedział, do którego należą wszystkie wartości średnie poszczególnych zmiennych, nie jest duży. ')
	st.image('final_scores_plot.png')

	st.write('Ostatni wykres przedstawia porównanie czterech modeli, w których podzielono zbiór danych na treningowy i testowy. Łatwo możemy zauważyć, że najgorzej dopasował się model DTR (Decision Tree Regression). Najlepszy natomiast okazał się model SVM (Support Vector Regression). Jeżeli dobrze przyjrzymy się wykresowi to możemy zauważyć, że model MLR (Multiple Linear Regression) również nie najgorzej dopasował się do naszych danych.')
	st.write('Poniżej przedstawione zostały wyniki poszczególnych modeli, których wizualizacja znajduje się na powyższym wykresie. Dzięki temu możemy zauważyć, że model SVM oraz MLR najlepiej dopasowały się do naszych danych. Ze wzgledu na nieco lepsze dopasowanie modelu SVM, wykorzystamy go w dalszej części projektu do przeprowadzenia predykcji.')

	df = pd.read_excel('final.xlsx')
	for row in df.values.tolist():
		st.write('Dla algorytmu ' + str(row[0])+'. Przy danych wejściowych typu '+str(row[1]) + ' dostaliśmy wynik R2 o wartości ' + str(round(row[2],2))) 
'''
