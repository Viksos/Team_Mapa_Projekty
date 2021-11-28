# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 17:59:40 2021

@author: Michał
"""

import pandas as pd
import numpy as np

class Cleaner():
    
    def __init__(self,df):
        self.df = df
    
    def How_many_nulls(self):
        return(print(self.df.isna().sum()))
    
    def Delete_rows_with_nulls(self):
        pass
    
    def Describe_plot(self,to_drop = 0 ):
        import seaborn as sns
        df = self.df
        if to_drop !=0 :
            df = df.drop(columns=[to_drop])
        import matplotlib.pyplot as plt
        figs = []
        for i in ['min','max','mean']:
            fig, ax = plt.subplots()
            ax = df.describe().loc[i].plot(kind = 'bar',stacked = False,color = np.array(sns.color_palette("rocket")))#color=plt.cm.magma(np.arange(len(df))))
            plt.title(i)
            figs.append(fig)
        return(figs)
            
    def Corelation_heatmap(self):
        import seaborn as sns
        data = self.df
        corr = data.corr()
        ax = sns.heatmap(
            corr, 
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        );
            
#df = pd.read_csv('Concrete_Data_Yeh.csv', ',')
#df1 = pd.read_csv('NFL Play by Play 2009-2016 (v3).csv',',',nrows = 10000)
#x = Cleaner(df1)
#x.How_many_nulls()