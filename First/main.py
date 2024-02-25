import yfinance as yf
import streamlit as st
import pandas as pd 

st.write("""# Simple Stock Price App
         
         Shown are the stockclosing price and volume of Google!

         """)

tickerSymbol = "AAPL"
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='id', start='2010-05-31', end='2020-05-31')

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)