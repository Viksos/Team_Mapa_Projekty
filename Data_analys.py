
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




def data_to_model(df):

	split_size = 0.33 

	predictions = pd.DataFrame() 
	
	X, y = df.drop(columns=['y']).to_numpy(), df['y'].to_numpy()

	scaler = StandardScaler()

	X = scaler.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = split_size, random_state=1)

	return(X_train, X_test, y_train, y_test)

<<<<<<< HEAD

def important_value(df):
	correlation = df.iloc[:,1:len(df.columns)].corr()
	return(correlation.tail(n=1).to_numpy()[0])	


def correlation(df):
	fig, ax = plt.subplots()
	corr = df.corr()
	sns.heatmap(corr,annot = True, cmap = 'rainbow')
	plt.title('Features Correlating ')
	return(fig)


=======
# heatmap of correlation

corr = df.iloc[:, 1:7].corr()
heatmap = sns.heatmap(corr[['y']].sort_values(by = 'y', ascending = False), vmin = 0, vmax = 1, annot = True, cmap = 'winter')
heatmap.set_title('Features Correlating with "y" variable', fontdict={'fontsize':14}, pad=15);
print(corr.iloc[[5]])

zmienne = corr.iloc[[5]].to_numpy()
zmienne = np.delete(zmienne, -1)
print(zmienne)

for i,v in enumerate(zmienne):
	print('Feature:', i,', Correlation:',v)

plt.bar([x for x in range(len(zmienne))], zmienne)
plt.ylabel("Correlation")
plt.xlabel("Features")
plt.show()

print(zmienne)

# Important features

important_features = []
for i in range(zmienne.size):
    if zmienne[i]>0.1:
        important_features.append(i)

print(important_features)
>>>>>>> 53dec890284664b42ef6adb97a55d96a09c1c8ce
