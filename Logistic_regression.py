# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:32:48 2024

@author: 06nic
"""

#Logit Algorithm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from itertools import chain , combinations
import matplotlib.pyplot as plt
import os
os.chdir(r"C:\Users\06nic\Portfolio_Management_Strategies\Risk_Parity")
import  codes.Risk_Parity_functions as rp

data_file = r"C:\Users\06nic\Portfolio_Management_Strategies\Risk_Parity\Data\Data.xlsx"


#Explanatory variables
CPI= pd.read_excel(data_file,sheet_name="CPI",header=0,index_col="Date")
CPI_MM = np.log(CPI).diff()
GDP_Growth= pd.read_excel(data_file,sheet_name="GDP_Growth",header=0,index_col="Date")
Real_estate= pd.read_excel(data_file,sheet_name="Real_estate_index",header=0,index_col="Date")
Real_estate_var=np.log(Real_estate).diff()
SOFR= pd.read_excel(data_file,sheet_name="SOFR",header=0,index_col="Date")
US10Y= pd.read_excel(data_file,sheet_name="US10Y",header=0,index_col="Date")
VIX= pd.read_excel(data_file,sheet_name="VIX",header=0,index_col="Date")
VIX_chng = np.log(VIX).diff()
SP500 = yf.download('^GSPC')["Adj Close"]
SP500_return = np.log(SP500).diff()
SP500_return = SP500_return.rename("SP500_return")
GOLD = yf.download('GC=F')["Adj Close"]
GOLD_return = np.log(GOLD).diff()
GOLD_return = GOLD_return.rename("GOLD_return")

all_variables  = [CPI_MM,GDP_Growth,Real_estate_var,SOFR,US10Y,VIX,SP500_return,GOLD_return]


# try to find which combinations have the best accuracy in prediction 
combi= chain.from_iterable(combinations(all_variables, s) for s in range(1,len(all_variables)+1))
combi = list(combi)



risk_parity_perf = rp.main(['^GSPC','GC=F',"BTC-USD"],"2001-01-01")
max_accuracy = 0


for variables in combi :
    variables = list(variables)
    data = pd.DataFrame()
    data.index = pd.date_range(start='2000-01-01', end='2024-04-27', freq='D')
    data.index.names = ['Date']
    for var in variables:
        data = pd.merge(data,var,on="Date",how="outer")
        
        
    data =data.resample("B").first()
    data = data.ffill()
    
    
    data= data.shift(2) # because we need to get data and invest the next day
    data = data.dropna()
    
    split_date = "2020-12-31"
    
    
    
    start_date = max([min(data.index),min(risk_parity_perf.index)])
    train_data = data[(data.index >=start_date)& (data.index <= split_date )]
    test_data = data[data.index >split_date ]
    
    
    
    train_perf_data = risk_parity_perf[(risk_parity_perf.index >= start_date )& (risk_parity_perf.index <= split_date )]
    train_perf_data = np.array([1 if i >0 else -1 for i in train_perf_data])
    test_perf_data = risk_parity_perf[risk_parity_perf.index >split_date ]
    test_perf_data = np.array([1 if i >0 else -1 for i in test_perf_data])
    
    
    
    #model 
    model = LogisticRegression()
    
    
    #train the model
    model.fit(np.array(train_data),train_perf_data)
    y_pred = model.predict(test_data)
    accuracy = accuracy_score(test_perf_data, y_pred)
    
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        store_data = data
        store_test_data = test_data
        store_model = model
        store_variables = data.columns
        
        
        
print("this variables", list(data.columns))
print("give an accuracy of", max_accuracy*100, "%")

y_pred = store_model.predict(store_test_data)
new_perf = risk_parity_perf[risk_parity_perf.index >split_date ] * y_pred


risk_parity_perf[risk_parity_perf.index >split_date ].cumsum().plot()
new_perf.cumsum().plot()
plt.legend(['Risk_parity_portfolio','Risk_parity_with_logistic_regression'])
plt.show()

rp.compute_ratios(new_perf)
rp.compute_ratios(risk_parity_perf[risk_parity_perf.index >split_date ])
