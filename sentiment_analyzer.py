"""
Sentiment Analysis Module for Stock News with Finnhub
NO PyTorch required - uses VADER sentiment analysis (lightweight & fast)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import re
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import VADER for sentiment analysis (no PyTorch needed!)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: vaderSentiment not installed. Using basic sentiment analysis.")
    print("Install with: pip install vaderSentiment")

# Try to import Finnhub
try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False
    print("Warning: finnhub-python not installed. Using alternative news sources.")
    print("Install with: pip install finnhub-python")


class SentimentAnalyzer:
    def __init__(self, finnhub_api_key=None):
        """
        Initialize sentiment analyzer with VADER (no PyTorch needed!)
        
        Args:
            finnhub_api_key: API key for Finnhub (get free at finnhub.io)
                           If None, will try to load from environment variable
        """
        # Try to get API key from parameter, then from .env file
        if finnhub_api_key is None:
            finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
        self.finnhub_api_key = finnhub_api_key
        
        # Initialize Finnhub client
        if finnhub_api_key and FINNHUB_AVAILABLE:
            try:
                self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)
                self.news_source = 'finnhub'
                print("✅ Finnhub client initialized successfully!")
            except Exception as e:
                print(f"Error initializing Finnhub: {e}")
                self.finnhub_client = None
                self.news_source = 'alternative'
        else:
            self.finnhub_client = None
            self.news_source = 'alternative'
            if finnhub_api_key and not FINNHUB_AVAILABLE:
                print("Finnhub API key provided but finnhub-python not installed.")
                print("Install with: pip install finnhub-python")
            elif not finnhub_api_key:
                print("⚠️ No Finnhub API key found.")
                print("Add FINNHUB_API_KEY to your .env file or get one at: https://finnhub.io/register")
        
        # Initialize VADER sentiment analyzer (lightweight, no ML models to download!)
        print("Initializing VADER sentiment analyzer...")
        if VADER_AVAILABLE:
            try:
                self.vader = SentimentIntensityAnalyzer()
                
                # Add financial-specific terms to VADER's lexicon
                financial_lexicon = {
                    'bullish': 2.5,
                    'bearish': -2.5,
                    'rally': 2.0,
                    'surge': 2.0,
                    'plunge': -2.5,
                    'crash': -3.0,
                    'soar': 2.5,
                    'tank': -2.5,
                    'beat': 2.0,
                    'miss': -2.0,
                    'upgrade': 2.0,
                    'downgrade': -2.0,
                    'outperform': 2.0,
                    'underperform': -2.0,
                    'buy': 1.5,
                    'sell': -1.5,
                    'hold': 0.0,
                    'profit': 1.5,
                    'loss': -1.5,
                    'gain': 1.5,
                    'decline': -1.5,
                    'growth': 1.5,
                    'recession': -2.5,
                    'bankruptcy': -3.0,
                    'merger': 1.0,
                    'acquisition': 1.0,
                    'layoffs': -2.0,
                    'earnings': 0.5,
                    'revenue': 0.5,
                    'dividend': 1.0
                }
                
                self.vader.lexicon.update(financial_lexicon)
                print("✅ VADER sentiment analyzer ready (with financial terms)!")
            except Exception as e:
                print(f"⚠️ Error initializing VADER: {e}")
                self.vader = None
        else:
            self.vader = None
            print("⚠️ VADER not available. Using basic sentiment analysis.")
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text using VADER
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if self.vader is not None:
            try:
                # Get VADER scores
                scores = self.vader.polarity_scores(text)
                
                return {
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'positive': scores['pos'],
                    'compound': scores['compound']  # Range: -1 to 1
                }
            except Exception as e:
                print(f"Error in VADER analysis: {e}")
                return self._simple_sentiment(text)
        else:
            return self._simple_sentiment(text)
    
    def _simple_sentiment(self, text):
        """Simple rule-based sentiment analysis fallback"""
        # Financial sentiment words with weights
        positive_words = {
            'good': 1, 'great': 2, 'excellent': 2, 'positive': 1, 'up': 1, 
            'gain': 1.5, 'profit': 2, 'growth': 2, 'rise': 1, 'surge': 2, 
            'rally': 2, 'bull': 1.5, 'bullish': 2, 'beat': 2, 'strong': 1.5, 
            'upgrade': 2, 'buy': 1.5, 'outperform': 2, 'soar': 2.5, 
            'record': 1.5, 'success': 2, 'innovative': 1.5, 'breakthrough': 2
        }
        
        negative_words = {
            'bad': -1, 'poor': -1.5, 'negative': -1, 'down': -1, 'loss': -2, 
            'decline': -1.5, 'fall': -1, 'drop': -1.5, 'crash': -3, 'bear': -1.5,
            'bearish': -2, 'concern': -1, 'risk': -1, 'miss': -2, 'weak': -1.5,
            'downgrade': -2, 'sell': -1.5, 'underperform': -2, 'plunge': -2.5,
            'tank': -2.5, 'bankruptcy': -3, 'layoffs': -2, 'lawsuit': -1.5,
            'investigation': -1.5, 'fraud': -3
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        pos_score = 0
        neg_score = 0
        
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            if clean_word in positive_words:
                pos_score += positive_words[clean_word]
            elif clean_word in negative_words:
                neg_score += abs(negative_words[clean_word])
        
        total = pos_score + neg_score
        
        if total == 0:
            return {
                'negative': 0.0,
                'neutral': 1.0,
                'positive': 0.0,
                'compound': 0.0
            }
        
        pos_ratio = pos_score / total
        neg_ratio = neg_score / total
        neutral_ratio = max(0, 1 - (pos_ratio + neg_ratio))
        
        # Compound score (normalized)
        compound = (pos_score - neg_score) / max(1, total)
        compound = max(-1, min(1, compound))  # Clamp between -1 and 1
        
        return {
            'negative': neg_ratio,
            'neutral': neutral_ratio,
            'positive': pos_ratio,
            'compound': compound
        }
    
    def fetch_news_finnhub(self, symbol, days=7):
        """
        Fetch news using Finnhub API
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for Finnhub (YYYY-MM-DD)
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            print(f"Fetching Finnhub news from {from_date} to {to_date}...")
            
            # Get company news
            news = self.finnhub_client.company_news(symbol, _from=from_date, to=to_date)
            
            print(f"✅ Fetched {len(news)} articles from Finnhub")
            
            # Convert to standard format
            articles = []
            for item in news:
                articles.append({
                    'title': item.get('headline', ''),
                    'description': item.get('summary', ''),
                    'publishedAt': datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'source': {'name': item.get('source', 'Finnhub')},
                    'url': item.get('url', ''),
                    'category': item.get('category', 'general'),
                    'sentiment': item.get('sentiment', None)
                })
            
            return articles
            
        except Exception as e:
            print(f"❌ Error fetching from Finnhub: {e}")
            return []
    
    def fetch_market_news_finnhub(self, category='general', max_articles=50):
        """
        Fetch general market news from Finnhub
        
        Args:
            category: News category ('general', 'forex', 'crypto', 'merger')
            max_articles: Maximum number of articles
            
        Returns:
            List of news articles
        """
        try:
            print(f"Fetching {category} market news from Finnhub...")
            
            # Get market news
            news = self.finnhub_client.general_news(category, minid=0)
            
            # Limit articles
            news = news[:max_articles]
            
            print(f"✅ Fetched {len(news)} market news articles")
            
            # Convert to standard format
            articles = []
            for item in news:
                articles.append({
                    'title': item.get('headline', ''),
                    'description': item.get('summary', ''),
                    'publishedAt': datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'source': {'name': item.get('source', 'Finnhub')},
                    'url': item.get('url', ''),
                    'category': item.get('category', category)
                })
            
            return articles
            
        except Exception as e:
            print(f"❌ Error fetching market news from Finnhub: {e}")
            return []
    
    def fetch_news_alternative(self, symbol, days=7):
        """
        Fetch news using free alternative sources (Google News RSS)
        Fallback when Finnhub is not available
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        articles = []
        
        print(f"Fetching news from Google News RSS (fallback)...")
        
        # Google News RSS (no API key needed)
        try:
            url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                
                for item in root.findall('.//item')[:50]:
                    title_elem = item.find('title')
                    pub_date_elem = item.find('pubDate')
                    link_elem = item.find('link')
                    
                    if title_elem is not None and pub_date_elem is not None:
                        try:
                            pub_date = datetime.strptime(
                                pub_date_elem.text, 
                                '%a, %d %b %Y %H:%M:%S %Z'
                            )
                        except:
                            pub_date = datetime.now()
                        
                        if (datetime.now() - pub_date).days <= days:
                            articles.append({
                                'title': title_elem.text,
                                'description': title_elem.text,
                                'publishedAt': pub_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                'source': {'name': 'Google News'},
                                'url': link_elem.text if link_elem is not None else ''
                            })
                
                print(f"✅ Fetched {len(articles)} articles from Google News")
                
        except Exception as e:
            print(f"❌ Error fetching from Google News RSS: {e}")
        
        return articles
    
    def fetch_news(self, symbol, days=7):
        """
        Fetch news articles for a stock symbol
        Tries Finnhub first, then falls back to alternatives
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        # Try Finnhub first if available
        if self.finnhub_client is not None:
            articles = self.fetch_news_finnhub(symbol, days)
            if articles:
                return articles
            print("⚠️ No articles from Finnhub, trying alternatives...")
        
        # Fallback to alternative sources
        articles = self.fetch_news_alternative(symbol, days)
        
        if not articles:
            print(f"⚠️ No news found for {symbol}. Using mock data for testing.")
            return self._get_mock_news(symbol, days)
        
        return articles
    
    def analyze_news_sentiment(self, symbol, days=7):
        """
        Fetch news and analyze sentiment for a stock
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            DataFrame with daily sentiment scores and article details
        """
        print(f"\n{'='*60}")
        print(f"📊 SENTIMENT ANALYSIS FOR {symbol}")
        print(f"{'='*60}\n")
        
        # Fetch articles
        articles = self.fetch_news(symbol, days)
        
        if not articles:
            print("⚠️ Using mock sentiment data for demonstration...")
            return self._get_mock_sentiment(days)
        
        print(f"\n🔍 Analyzing {len(articles)} articles...\n")
        
        # Analyze each article
        sentiment_data = []
        
        for i, article in enumerate(articles):
            try:
                # Combine title and description
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                # Clean text
                text = self._clean_text(text)
                
                if len(text) < 10:  # Skip very short texts
                    continue
                
                # Analyze sentiment
                sentiment = self.analyze_text(text)
                
                # Parse date - FIX: Convert to pandas datetime immediately
                try:
                    pub_datetime = datetime.strptime(
                        article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'
                    )
                    pub_date = pd.to_datetime(pub_datetime.date())
                except:
                    pub_date = pd.to_datetime(datetime.now().date())
                
                sentiment_data.append({
                    'date': pub_date,
                    'title': article.get('title', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', ''),
                    'sentiment_score': sentiment['compound'],
                    'positive': sentiment['positive'],
                    'negative': sentiment['negative'],
                    'neutral': sentiment['neutral'],
                    'category': article.get('category', 'general')
                })
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  ✓ Processed {i + 1}/{len(articles)} articles...")
            
            except Exception as e:
                print(f"  ⚠️ Error analyzing article: {e}")
                continue
        
        if not sentiment_data:
            print("⚠️ No sentiment data generated. Using mock data.")
            return self._get_mock_sentiment(days)
        
        # Create DataFrame
        df = pd.DataFrame(sentiment_data)
        
        print(f"\n✅ Analysis complete! Processed {len(sentiment_data)} articles\n")
        
        # Aggregate by date
        daily_sentiment = df.groupby('date').agg({
            'sentiment_score': 'mean',
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'title': 'count'
        }).reset_index()
        
        daily_sentiment.columns = ['date', 'sentiment_score', 'positive_avg', 
                                   'negative_avg', 'neutral_avg', 'news_count']
        
        # Fill missing dates - FIX: Ensure date_range is also pandas datetime
        date_range = pd.date_range(
            end=datetime.now().date(),
            periods=days,
            freq='D'
        )
        
        # Create full date range DataFrame with proper datetime type
        full_df = pd.DataFrame({'date': date_range})
        
        # Merge with daily sentiment
        full_df = full_df.merge(daily_sentiment, on='date', how='left')
        full_df = full_df.fillna(0)
        
        # Print summary
        avg_sentiment = full_df['sentiment_score'].mean()
        print(f"📈 Average Sentiment Score: {avg_sentiment:.3f}")
        
        if avg_sentiment > 0.1:
            print(f"   → Bullish sentiment 🐂")
        elif avg_sentiment < -0.1:
            print(f"   → Bearish sentiment 🐻")
        else:
            print(f"   → Neutral sentiment ➡️")
        
        print(f"\n{'='*60}\n")
        
        return full_df
    
    def get_detailed_articles(self, symbol, days=7, top_n=10):
        """
        Get detailed article information with sentiment scores
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            top_n: Number of top articles to return
            
        Returns:
            DataFrame with article details sorted by absolute sentiment
        """
        articles = self.fetch_news(symbol, days)
        
        if not articles:
            return pd.DataFrame()
        
        article_details = []
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            text = self._clean_text(text)
            
            if len(text) < 10:
                continue
            
            sentiment = self.analyze_text(text)
            
            try:
                pub_date = datetime.strptime(
                    article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'
                )
            except:
                pub_date = datetime.now()
            
            article_details.append({
                'date': pub_date,
                'title': article.get('title', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'sentiment_score': sentiment['compound'],
                'sentiment_label': self._get_sentiment_label(sentiment['compound']),
                'url': article.get('url', '')
            })
        
        df = pd.DataFrame(article_details)
        
        if df.empty:
            return df
        
        # Sort by absolute sentiment (most extreme first)
        df['abs_sentiment'] = df['sentiment_score'].abs()
        df = df.sort_values('abs_sentiment', ascending=False)
        df = df.drop('abs_sentiment', axis=1)
        
        return df.head(top_n)
    
    def _get_sentiment_label(self, score):
        """Convert sentiment score to label"""
        if score > 0.5:
            return "Very Positive 🚀"
        elif score > 0.1:
            return "Positive ⬆️"
        elif score < -0.5:
            return "Very Negative 📉"
        elif score < -0.1:
            return "Negative ⬇️"
        else:
            return "Neutral ➡️"
    
    def _clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters but keep basic punctuation for VADER
        text = re.sub(r'[^\w\s!?.,-]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _get_mock_news(self, symbol, days):
        """Generate mock news data for testing"""
        mock_titles = [
            f"{symbol} reports strong quarterly earnings beat",
            f"Analysts upgrade {symbol} stock rating to buy",
            f"{symbol} announces innovative new product launch",
            f"Market volatility impacts {symbol} trading volume",
            f"{symbol} CEO outlines ambitious growth strategy",
            f"Institutional investors increase {symbol} holdings",
            f"{symbol} faces regulatory scrutiny over practices",
            f"Technical analysis shows bullish trend for {symbol}",
            f"{symbol} stock price reaches new 52-week high",
            f"Concerns raised about {symbol} competitive position"
        ]
        
        articles = []
        for i in range(min(days * 3, 30)):
            articles.append({
                'title': mock_titles[i % len(mock_titles)],
                'description': f"Mock article about {symbol} for testing purposes",
                'publishedAt': (datetime.now() - timedelta(days=i//3)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'source': {'name': 'Mock Financial News'},
                'url': f'https://example.com/article-{i}',
                'category': 'company'
            })
        
        return articles
    
    def _get_mock_sentiment(self, days):
        """Generate mock sentiment data"""
        import numpy as np
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate somewhat realistic sentiment trends
        base_sentiment = np.random.uniform(-0.2, 0.2)
        trend = np.random.uniform(-0.05, 0.05, days).cumsum()
        noise = np.random.uniform(-0.1, 0.1, days)
        
        sentiment_scores = np.clip(base_sentiment + trend + noise, -1, 1)
        
        return pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'positive_avg': np.random.uniform(0.3, 0.6, days),
            'negative_avg': np.random.uniform(0.1, 0.3, days),
            'neutral_avg': np.random.uniform(0.2, 0.5, days),
            'news_count': np.random.randint(5, 30, days)
        })
    
    def get_sentiment_summary(self, df):
        """
        Get summary statistics of sentiment
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty or len(df) == 0:
            return {
                'avg_sentiment': 0.0,
                'recent_sentiment': 0.0,
                'sentiment_trend': 'unknown',
                'total_articles': 0,
                'most_positive_day': None,
                'most_negative_day': None,
                'sentiment_volatility': 0.0,
                'days_analyzed': 0
            }
        
        recent_sentiment = df['sentiment_score'].iloc[-3:].mean()
        overall_sentiment = df['sentiment_score'].mean()
        
        return {
            'avg_sentiment': overall_sentiment,
            'recent_sentiment': recent_sentiment,
            'sentiment_trend': 'improving' if recent_sentiment > overall_sentiment else 'declining',
            'total_articles': int(df['news_count'].sum()),
            'most_positive_day': df.loc[df['sentiment_score'].idxmax(), 'date'],
            'most_negative_day': df.loc[df['sentiment_score'].idxmin(), 'date'],
            'sentiment_volatility': df['sentiment_score'].std(),
            'days_analyzed': len(df)
        }


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STOCK SENTIMENT ANALYZER (No PyTorch Required!)")
    print("="*60 + "\n")
    
    # Initialize analyzer (automatically loads from .env)
    analyzer = SentimentAnalyzer()
    
    # Analyze sentiment for a stock
    symbol = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT) [default: AAPL]: ").strip().upper()
    
    if not symbol:
        symbol = "AAPL"
        print(f"Using default symbol: {symbol}")
    
    days = 7
    
    # Get sentiment analysis
    sentiment_df = analyzer.analyze_news_sentiment(symbol, days=days)
    
    print(f"\n📊 Daily Sentiment Scores:")
    print(sentiment_df.to_string(index=False))
    
    # Get summary
    summary = analyzer.get_sentiment_summary(sentiment_df)
    print(f"\n📈 Sentiment Summary:")
    print("-" * 60)
    for key, value in summary.items():
        print(f"{key:.<25} {value}")
    
    # Get detailed articles
    print(f"\n📰 Top Articles with Sentiment Scores:")
    print("-" * 60)
    detailed_articles = analyzer.get_detailed_articles(symbol, days=days, top_n=5)
    if not detailed_articles.empty:
        for idx, row in detailed_articles.iterrows():
            print(f"\n{row['sentiment_label']}")
            print(f"Title: {row['title'][:80]}...")
            print(f"Source: {row['source']} | Score: {row['sentiment_score']:.3f}")
            if row['url']:
                print(f"URL: {row['url'][:60]}...")
    
    # Test single text analysis
    print(f"\n\n🧪 Testing Single Text Analysis:")
    print("-" * 60)
    sample_texts = [
        f"{symbol} surges on strong earnings report, beating all estimates",
        f"{symbol} stock plummets amid regulatory concerns and weak guidance",
        f"{symbol} maintains steady performance in volatile market conditions"
    ]
    
    for text in sample_texts:
        result = analyzer.analyze_text(text)
        label = analyzer._get_sentiment_label(result['compound'])
        print(f"\nText: {text}")
        print(f"Sentiment: {label} (Score: {result['compound']:.3f})")
        print(f"Positive: {result['positive']:.2f} | Negative: {result['negative']:.2f} | Neutral: {result['neutral']:.2f}")
    
    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print("="*60 + "\n")
