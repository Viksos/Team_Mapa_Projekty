import pandas as pd
import streamlit as st
import Inputs, Data_analys

from io import BytesIO
from pyxlsb import open_workbook as open_xlsb


# writer = pd.ExcelWriter('pandas_multiple.xlsx')

# df.to_excel(writer,sheet_name = 'Data')
##X_train.to_excel(writer,sheet_name = 'Training X')
# y_train.to_excel(writer,sheet_name ='Training Y')
# X_test.to_excel(writer,sheet_name = 'Test X')
# y_test.to_excel(writer,sheet_name ='Test Y')
# writer.save()

def to_excel(df):
	output = BytesIO()
	writer = pd.ExcelWriter(output)
	df.to_excel(writer, index=False, sheet_name='Sheet1')
	workbook = writer.book
	worksheet = writer.sheets['Sheet1']
	# format1 = workbook.add_format({'num_format': '0.00'})
	# worksheet.set_column('A:A', None, format1)
	writer.save()
	processed_data = output.getvalue()
	return processed_data


def app():  # Narazie tylko pobieranie excela z danymi XD

	df = Inputs.return_df()
	# X_train, X_test, y_train, y_test = Data_analys.data_to_model(df)

	type_of_file = st.selectbox('Choose data to download :', ['Raw Data'])  # ,'Train','Test'])

	if (type_of_file == 'Raw Data'):
		df_xlsx = to_excel(df)

	# elif(type_of_file == 'Train'):
	#	df_xlsx = to_excel(X_train)

	# elif(type_of_file == 'Test'):
	#	df_xlsx = to_excel(X_test)

	st.download_button(label='Download',
					   data=df_xlsx,
					   file_name='df_test.xlsx')
