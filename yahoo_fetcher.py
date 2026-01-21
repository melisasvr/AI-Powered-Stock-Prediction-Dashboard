import yfinance as yf
import pandas as pd
import numpy as np
import time

class YahooDataFetcher:
    def __init__(self):
        print("Yahoo fetcher started (simple version)")

    def get_historical_data(self, symbol, days=365):
        for attempt in range(6):
            try:
                ticker = yf.Ticker(symbol)
                # Try to get 2 years, then cut to what user wants
                df = ticker.history(period="2y", interval="1d")

                if df.empty:
                    print(f"Attempt {attempt+1}: No data received for {symbol}")
                    time.sleep(4)
                    continue

                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

                # Keep only the last X days the user asked for
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
                df = df[df['date'] > cutoff]

                if df.empty:
                    print(f"No data in the last {days} days for {symbol}")
                    return None

                print(f"Got {len(df)} days of data for {symbol}")
                return df

            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(5)

        print(f"Could not get data for {symbol} after several tries.")
        return None

    def calculate_technical_indicators(self, df):
        if df is None or df.empty:
            return df
        df = df.copy()

        df['sma_50'] = df['close'].rolling(window=min(50, len(df))).mean()
        df['sma_200'] = df['close'].rolling(window=min(200, len(df))).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']

        return df.fillna(method='bfill').fillna(0)