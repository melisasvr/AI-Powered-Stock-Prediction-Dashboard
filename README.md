# 📈 AI-Powered Stock Prediction Dashboard

A lightweight, free-to-use stock market analysis and prediction tool built with Python. Features machine learning price predictions, sentiment analysis from news, and technical indicators visualization, all without requiring paid API keys!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## ✨ Features

- 📊 **Real-time Stock Data** - Free data from Yahoo Finance (no API key needed)
- 🤖 **ML Price Predictions** - Random Forest, Gradient Boosting, and Linear Regression models
- 💭 **Sentiment Analysis** - VADER-based news sentiment (lightweight, no PyTorch)
- 📈 **Technical Indicators** - SMA, EMA, RSI, MACD, Bollinger Bands
- 📰 **News Integration** - Finnhub API support + Google News fallback
- 🎯 **Confidence Intervals** - Uncertainty estimation for predictions
- 🚀 **Fast & Lightweight** - No heavy ML frameworks required

## 🎥 Demo

```bash
# Run the dashboard
streamlit run app.py
```

Then enter a stock ticker (e.g., AAPL, TSLA, MSFT) and click "Run"!

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🚀 Quick Start

### 1. Clone or Download

```bash
git clone <your-repo-url>
cd stock-prediction-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Minimum requirements:**
```bash
pip install streamlit yfinance scikit-learn pandas plotly vaderSentiment
```

### 3. (Optional) Set Up API Keys

Create a `.env` file in the project root:

```env
# Optional: For enhanced news sentiment analysis
FINNHUB_API_KEY=your_finnhub_api_key_here
```

Get a free Finnhub API key at: https://finnhub.io/register

**Note:** The app works without any API keys using Google News RSS!

### 4. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
stock-prediction-dashboard/
│
├── app.py                     # Main Streamlit application
├── yahoo_fetcher.py          # Yahoo Finance data fetcher
├── model_trainer.py          # ML models (Random Forest, etc.)
├── sentiment_analyzer.py     # News sentiment analysis (VADER)
├── requirements.txt          # Python dependencies
├── .env                      # API keys (optional, not committed)
├── .env.example              # Example environment variables
└── README.md                 # This file
```

## 📦 Installation Guide

### Full Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

### Minimal Installation (No News Sentiment)

```bash
pip install streamlit yfinance scikit-learn pandas plotly
```

### requirements.txt

```txt
streamlit==1.29.0
pandas==2.1.3
numpy==1.24.3
yfinance==0.2.32
scikit-learn==1.3.2
plotly==5.18.0
matplotlib==3.8.2
vaderSentiment==3.3.2
python-dotenv==1.0.0

# Optional: For enhanced sentiment analysis
finnhub-python==2.4.19
requests==2.31.0
beautifulsoup4==4.12.2
lxml==4.9.3
```

## 🎮 Usage

### Basic Usage

1. **Enter Stock Symbol**: Type any ticker (AAPL, TSLA, GOOGL, etc.)
2. **Select History**: Choose how many days of historical data to analyze
3. **Click Run**: Get predictions and analysis!

### Advanced Features

#### In app.py

```python
from yahoo_fetcher import YahooDataFetcher
from model_trainer import StockPredictor
from sentiment_analyzer import SentimentAnalyzer

# Fetch stock data
fetcher = YahooDataFetcher()
df = fetcher.get_historical_data('AAPL', days=365)
df = fetcher.calculate_technical_indicators(df)

# Train model and predict
predictor = StockPredictor(model_type='random_forest')
predictor.train(df)
predictions = predictor.predict(df, days_ahead=7)

# Get sentiment analysis
analyzer = SentimentAnalyzer()
sentiment_df = analyzer.analyze_news_sentiment('AAPL', days=7)
```

### Model Options

Choose from three ML models:

- **Random Forest** (Best accuracy, provides confidence intervals)
- **Gradient Boosting** (Fast and accurate)
- **Linear Regression** (Fastest, simple trends)

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Optional: Finnhub API for better news coverage
FINNHUB_API_KEY=your_key_here
```

### Customizing Models

Edit `model_trainer.py` to adjust model parameters:

```python
# In StockPredictor.__init__()
if model_type == 'random_forest':
    self.model = RandomForestRegressor(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Maximum depth
        min_samples_split=5,   # Min samples to split
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
```

## 📊 Available Technical Indicators

- **Moving Averages**: SMA-50, SMA-200
- **Momentum**: RSI (Relative Strength Index)
- **Trend**: EMA-12, EMA-26, MACD

## 🔍 Sentiment Analysis

### News Sources

1. **Finnhub** (if API key provided) - Best coverage
2. **Google News RSS** - Free fallback, no key needed
3. **Mock Data** - For testing when offline

### Sentiment Scoring

- Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Enhanced with financial-specific terms
- Scores range from -1 (very negative) to +1 (very positive)

## ⚠️ Known Limitations

1. **Yahoo Finance Rate Limits**: 
   - May limit requests if too frequent
   - Solution: Wait 30-60 seconds between queries
   - Data is cached for 10 minutes

2. **Prediction Accuracy**: 
   - ML predictions are estimates, not guarantees
   - Market conditions change rapidly
   - Always verify with other sources

3. **News Coverage**: 
   - Free sources have limited articles
   - Consider Finnhub API for better coverage

## 🐛 Troubleshooting

### "Too Many Requests" Error

**Problem**: Yahoo Finance rate limited your IP

**Solutions**:
- Wait 30-60 seconds and try again
- Reduce the lookback period (use 90 days instead of 365)
- Data is cached - avoid repeated requests for same symbol

### "No data found" Error

**Problem**: Invalid ticker symbol or no data available

**Solutions**:
- Verify the ticker symbol is correct
- Try a major stock (AAPL, MSFT, GOOGL)
- Check if market is open

### Import Errors

**Problem**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

### Prediction Errors

**Problem**: Not enough historical data

**Solutions**:
- Increase the "Days of history" slider
- Use at least 90 days for better predictions
- Check if stock has sufficient trading history

## 🚀 Performance Tips

1. **Cache Data**: The app caches data for 1 hour - reuse queries
2. **Batch Analysis**: Analyze multiple stocks by saving results
3. **Model Selection**: Use Linear Regression for fastest predictions
4. **Data Range**: 90-180 days is usually sufficient for good predictions

## 📈 Example Output

```
Stock: AAPL
Last close: $178.45

Predictions for next 7 days:
Day 1: $179.12 (+0.38%)
Day 2: $179.85 (+0.78%)
Day 3: $180.23 (+1.00%)
Day 4: $180.67 (+1.24%)
Day 5: $181.02 (+1.44%)
Day 6: $181.34 (+1.62%)
Day 7: $181.58 (+1.75%)

Sentiment Analysis:
Average sentiment: 0.234 (Positive 📈)
Total articles analyzed: 42
Trend: Bullish 🐂
```

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions

- [ ] Add more technical indicators
- [ ] Support for cryptocurrency data
- [ ] Portfolio tracking features
- [ ] Backtesting functionality
- [ ] Export predictions to CSV/Excel
- [ ] Email alerts for price movements
- [ ] Dark mode UI
- [ ] Multi-language support

## 📝 License

This project is licensed under the MIT License - see below:

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ⚠️ Disclaimer

**THIS SOFTWARE IS FOR EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY.**
- Not financial advice
- No guarantee of accuracy
- Past performance doesn't predict future results
- Always do your own research (DYOR)
- Consult a licensed financial advisor for investment decisions
- The creators are not responsible for any financial losses

## 📚 Resources
- [Yahoo Finance](https://finance.yahoo.com/) - Stock data source
- [yfinance Documentation](https://pypi.org/project/yfinance/) - Python library
- [Finnhub API](https://finnhub.io/) - News and market data
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) - Sentiment analysis
- [Streamlit Docs](https://docs.streamlit.io/) - Dashboard framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning library

## 🙋 FAQ

**Q: Do I need API keys to use this?**  
A: No! The app works completely free with Yahoo Finance and Google News RSS.

**Q: How accurate are the predictions?**  
A: Accuracy varies. Use predictions as one of many tools, not the sole decision-maker.

**Q: Can I use this for crypto or forex?**  
A: Currently only stocks. Crypto/forex support could be added as a feature.

**Q: Is my data secure?**  
A: All data is processed locally. No data is sent to third parties (except API calls).

**Q: Can I run this on a server?**  
A: Yes! Deploy to Streamlit Cloud, Heroku, or any Python hosting service.

**Q: How do I update to the latest version?**  
A: Pull the latest changes and run `pip install -r requirements.txt --upgrade.`

## 📧 Contact & Support
- **Issues**: Open an issue on GitHub

## 🌟 Acknowledgments
- Yahoo Finance for free stock data
- Finnhub for news API
- Streamlit for the amazing framework
- The open-source community

---

**Made with ❤️ for the trading community**

**Star ⭐ this repo if you find it useful!**
