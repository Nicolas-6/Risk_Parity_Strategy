# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:11:14 2024

@author: 06nic
"""
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import re
import matplotlib.pyplot as plt


today_date = dt.date.today().strftime("%Y-%m-%d")

def download_data(tickers:list,start_date,end_date=today_date):
    pattern = r"^\d{4}-\d{1,2}-\d{1,2}$"
    if not (re.match(pattern,start_date) ) or not(re.match(pattern,end_date)):
        raise ValueError("Please enter your start_date and end_date in this format 'YYYY-MM-DD'")
    else:
        data = yf.download(tickers, start=start_date, end=end_date)
        data = data["Adj Close"]
        return data

def compute_return (data):
    data_return = np.log(data).diff()
    return data_return
    
def compute_historical_moving_volatility(data_return,window):
    data_volatility= data_return.rolling(window).std()
    return data_volatility

def risk_parity_portfolio_weigths(data_volatility):
    inverse_vol = 1/data_volatility
    sum_inverse = inverse_vol.sum(axis=1)
    weigths= inverse_vol.divide(sum_inverse,axis=0)
    return weigths

def risk_parity_performance(data_return,weights,shift=2):
    performance = data_return * weights.shift(shift)
    performance_portf =  performance.sum(axis=1)
    performance_portf = performance_portf.rename("Risk Parity Portfolio")

    return performance_portf

def compute_drawdown(data_return):
    cum_sum = data_return.cumsum()
    drawdowns = cum_sum.cummax() - cum_sum

    return drawdowns
    
    
def compute_ratios(data_return): #data_return expected dim(1,N)
    if data_return.ndim != 1: 
        raise TypeError ("Please this function accepts only 1 dimension variable")
    else :
        
        total_return = sum(data_return)
        std = np.std(data_return.dropna())
        drawdowns =compute_drawdown(data_return)
        max_drawdown = max(drawdowns)
        
        data_return.cumsum().plot()
        plt.title("Risk Parity Portfolio Performance")
        plt.show()
        drawdowns.plot()
        plt.title("Historical Drawdowns")
        plt.show()
        
        print("Return:",round(total_return*100,2),"%")
        print("Risk (Standard deviation):",round(std*100,2),"%")
        print("Max Drawdown :",round(max_drawdown*100,2),"%" )
    
data = download_data(['^GSPC','GC=F',"BTC-USD"],"2001-01-01")
data = data.resample("B").first().ffill()
data=data.dropna()
returns = compute_return(data)
vol = compute_historical_moving_volatility(returns,5)
weights = risk_parity_portfolio_weigths(vol)
perf = risk_parity_performance(returns,weights)

def main(tickers, start_date,end_date=today_date,window=5, shift=2):
    data = download_data(tickers,start_date,end_date)
    data = data.resample("B").first().ffill()
    data=data.dropna()
    returns = compute_return(data)
    vol = compute_historical_moving_volatility(returns,window)
    weights = risk_parity_portfolio_weigths(vol)
    perf = risk_parity_performance(returns,weights,shift)
    return perf [window+shift :]
    
    
    
drawdowns = compute_drawdown(returns)

compute_ratios(perf)