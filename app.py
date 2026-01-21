import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os

load_dotenv()

from yahoo_fetcher import YahooDataFetcher
from model_trainer import StockPredictor
from sentiment_analyzer import SentimentAnalyzer

@st.cache_data(ttl=3600)
def get_stock_data(symbol, days):
    fetcher = YahooDataFetcher()
    data = fetcher.get_historical_data(symbol, days)
    if data is not None:
        return fetcher.calculate_technical_indicators(data)
    return None

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("Stock Price Predictor")

symbol = st.sidebar.text_input("Ticker", "AAPL").upper()
days = st.sidebar.slider("Days of history", 60, 730, 365)

if st.sidebar.button("Run"):
    df = get_stock_data(symbol, days)

    if df is None:
        st.error(f"Could not get data for {symbol}. Try again later or different ticker.")
    else:
        try:
            predictor = StockPredictor(model_type="random_forest")
            predictor.train(df)
            preds = predictor.predict(df, days_ahead=7)
        except Exception as e:
            st.warning("Prediction failed → showing only history")
            preds = [df['close'].iloc[-1]] * 7

        st.write(f"Last close: ${df['close'].iloc[-1]:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['close'], name="Price"))
        
        last_date = df['date'].iloc[-1]
        future = [last_date + timedelta(days=i) for i in range(1, 8)]
        fig.add_trace(go.Scatter(x=future, y=preds, name="Prediction", line=dict(dash='dot')))

        st.plotly_chart(fig)