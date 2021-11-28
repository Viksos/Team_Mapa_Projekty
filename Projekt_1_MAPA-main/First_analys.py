import streamlit as st
import Clean as Cl
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def app():

	st.set_option('deprecation.showPyplotGlobalUse', False)
	df = pd.read_csv('dane.csv', ',').iloc[:,1:]

	spr_df = Cl.Cleaner(df)

	st.write('Na początku przeanalizujmy wykres korelacji pomiędzy zmiennymi.')

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

