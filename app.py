# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:24:24 2024

@author: 06nic
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
import matplotlib.pyplot as plt

# Define the functions
def download_data(tickers, start_date, end_date):
    pattern = r"^\d{4}-\d{1,2}-\d{1,2}$"
    if not (re.match(pattern, start_date)) or not (re.match(pattern, end_date)):
        raise ValueError("Please enter your start_date and end_date in this format 'YYYY-MM-DD'")
    else:
        data = yf.download(tickers, start=start_date, end=end_date)
        data = data["Adj Close"]
        return data

def compute_return(data):
    data_return = np.log(data).diff()
    return data_return

def compute_historical_moving_volatility(data_return, window):
    data_volatility = data_return.rolling(window).std()
    return data_volatility

def risk_parity_portfolio_weights(data_volatility):
    inverse_vol = 1 / data_volatility
    sum_inverse = inverse_vol.sum(axis=1)
    weights = inverse_vol.divide(sum_inverse, axis=0)
    return weights

def risk_parity_performance(data_return, weights, shift=2):
    data_return = data_return[weights.index[0]:]
    performance = data_return * weights.shift(shift)
    performance_portf = performance.sum(axis=1)
    performance_portf = performance_portf.rename("Risk Parity Portfolio")
    return performance_portf

def risk_parity_portfolio(tickers, start_date, end_date, window=5, shift=2, rebalancing="M"):
    data = download_data(tickers, start_date, end_date)
    data = data.resample(rebalancing).first().ffill()
    data = data.dropna()
    returns = compute_return(data)
    vol = compute_historical_moving_volatility(returns, window)
    weights = risk_parity_portfolio_weights(vol)
    perf = risk_parity_performance(returns, weights, shift)
    return perf[window + shift:]


def compute_drawdown(data_return):
    cum_sum = np.exp(data_return.cumsum())
    cum_max = cum_sum.cummax()
    drawdowns = (cum_max - cum_sum) / cum_max
    return drawdowns

# Streamlit app
st.title('Risk Parity Portfolio')

st.sidebar.header('Input Parameters')

# User inputs
tickers = st.sidebar.text_input('Tickers (comma separated)', 'AAPL,MSFT,GOOGL')
start_date = st.sidebar.text_input('Start Date (YYYY-MM-DD)', '2020-01-01')
end_date = st.sidebar.text_input('End Date (YYYY-MM-DD)', pd.Timestamp.today().strftime('%Y-%m-%d'))
window = st.sidebar.slider('Window used for rebalancing', 1, 252, 5)
shift = st.sidebar.slider('Shift', 1, 10, 2)
rebalancing = st.sidebar.selectbox('Rebalancing Frequency', ['B', 'W', 'M'])

# Convert tickers to list
tickers_list = tickers.split(',')

# Run the risk parity portfolio function and display the results
if st.sidebar.button('Run'):
    try:
        performance = risk_parity_portfolio(tickers_list, start_date, end_date, window, shift, rebalancing)
        cumulative_performance = performance.cumsum()
        
        # Compute drawdowns
        drawdowns = compute_drawdown(performance)
        
        
        st.header('Risk Parity Portfolio Performance')
        #st.line_chart(cumulative_performance)
        
        # Plotting using matplotlib for cumulative performance
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots()
        cumulative_performance.plot(ax=ax)
        ax.set_title('Risk Parity Portfolio Performance')
        ax.grid(True)
        ax.set_ylabel('Cumulative Returns')
        ax.set_xlabel('Date')
        st.pyplot(fig)
        
        st.header('Risk Parity Portfolio Drawdowns')
        #st.line_chart(drawdowns)
        
        # Plotting using matplotlib for drawdowns
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots()
        drawdowns.plot(ax=ax,color='red')
        ax.set_title('Risk Parity Portfolio Drawdowns')
        ax.grid(True)
        ax.set_ylabel('Drawdown')
        ax.set_xlabel('Date')
        st.pyplot(fig)

        # Right sidebar for statistics
        st.sidebar.header('Statistics')
        
        mean_return = performance.mean()* 252
        std_dev = performance.std() * np.sqrt(252)
        sharpe_ratio = mean_return / std_dev
        max_drawdown = drawdowns.max()

        st.sidebar.write(f"Mean Return: {mean_return:.4f}")
        st.sidebar.write(f"Standard Deviation (Risk): {std_dev:.4f}")
        st.sidebar.write(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        st.sidebar.write(f"Max Drawdown: {max_drawdown:.4f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")